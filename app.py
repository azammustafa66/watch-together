"""Streamlit app for individual and group movie recommendations.

Loads an existing local model, collects a few explicit ratings per viewer,
and fits temporary viewer vectors. No external API or streaming service is used.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch

from watch_together.data import digest
from watch_together.recommend import fit_viewer, load_recommender, rank_movies

ROOT = Path(__file__).resolve().parent


@st.cache_resource
def resources(model_hash: str, catalog_hash: str):
    """Cache a frozen scorer and catalog, invalidating on artifact content changes."""
    torch.set_num_threads(4)
    return load_recommender(ROOT)


def seed_demo(catalog: pd.DataFrame) -> None:
    """Populate editable example ratings for three viewers with different tastes."""
    examples = [
        [('Matrix, The (1999)', 5.0), ('Toy Story (1995)', 4.0), ('Titanic (1997)', 2.0)],
        [('Toy Story (1995)', 5.0), ('Titanic (1997)', 4.5), ('Matrix, The (1999)', 3.0)],
        [('Jurassic Park (1993)', 5.0), ('Matrix, The (1999)', 4.5), ('Titanic (1997)', 2.5)],
    ]
    for member, pairs in enumerate(examples):
        choices = []
        for title, rating in pairs:
            matches = catalog.loc[catalog.title == title, 'item_idx']
            if len(matches):
                item = int(matches.iloc[0])
                choices.append(item)
                st.session_state[f'rating_{member}_{item}'] = rating
        st.session_state[f'choices_{member}'] = choices
    st.session_state.pop('recommendations', None)


def main() -> None:
    """Render preference entry, ranking controls, per-viewer scores, and evaluation."""
    st.set_page_config(page_title='Watch Together', page_icon='🎬', layout='wide')
    st.title('What should we watch together?')
    st.write('Rate a few movies. Find a shortlist for yourself—or a compromise for movie night.')
    st.caption(
        'Historical MovieLens Small catalog · local recommendations · '
        'no current releases or streaming availability'
    )
    required = ['selected.pt', 'catalog.csv']
    if any(not (ROOT / 'artifacts' / name).exists() for name in required):
        st.info('Prepare the dataset and train the models first:')
        st.code('uv run watch-together all', language='bash')
        st.stop()
    try:
        bundle = resources(
            digest(ROOT / 'artifacts/selected.pt'), digest(ROOT / 'artifacts/catalog.csv')
        )
    except (ValueError, OSError) as error:
        st.error(str(error))
        st.stop()
    catalog = bundle.catalog
    mode = st.radio('Who is watching?', ['Individual', 'Group'], horizontal=True)
    members = st.select_slider('Group size', options=[2, 3], value=2) if mode == 'Group' else 1
    st.button('Try example ratings', on_click=seed_demo, args=(catalog,))
    st.subheader('Your movie preferences')
    st.caption('Choose at least three movies per person. Include likes and dislikes when possible.')
    options = catalog.sort_values(['count', 'movieId'], ascending=[False, True]).item_idx.tolist()
    profiles = []
    for member, column in enumerate(st.columns(members)):
        with column:
            st.markdown(f'**Viewer {member + 1}**')
            chosen = st.multiselect(
                'Search movies you have watched',
                options,
                key=f'choices_{member}',
                format_func=lambda item: catalog.iloc[item].title,
                max_selections=12,
            )
            ratings = {}
            for item in chosen:
                rating_key = f'rating_{member}_{item}'
                st.session_state.setdefault(rating_key, 3.5)
                ratings[int(item)] = st.slider(
                    catalog.iloc[item].title,
                    0.5,
                    5.0,
                    None,
                    0.5,
                    key=rating_key,
                )
            profiles.append(ratings)

    with st.expander('Tune the shortlist', expanded=True):
        left, right = st.columns(2)
        with left:
            available_genres = sorted(
                {
                    genre
                    for value in catalog.genres
                    for genre in value.split('|')
                    if genre != '(no genres listed)'
                }
            )
            genres = st.multiselect('Genres (match any)', available_genres)
            year_filter = st.checkbox('Filter release years')
            years = None
            if year_filter:
                lower, upper = int(catalog.year.min()), int(catalog.year.max())
                years = st.slider('Release years', lower, upper, (lower, upper))
            minimum = st.slider('Minimum historical ratings per movie', 1, 50, 10)
        with right:
            compromise = (
                st.slider(
                    'Protect the least satisfied viewer',
                    0,
                    100,
                    50,
                    help='0 uses average enjoyment; 100 uses the lowest viewer prediction.',
                    disabled=members == 1,
                )
                / 100
            )
            discovery = (
                st.slider(
                    'Discovery: familiar → less popular',
                    0,
                    100,
                    0,
                    help='At maximum, historical rarity contributes 30% of the ranking score.',
                )
                / 100
            )
            method = st.selectbox('Recommendation method', ['Personalized', 'Popularity baseline'])
    signature = json.dumps(
        {
            'profiles': profiles,
            'genres': genres,
            'years': years,
            'minimum': minimum,
            'compromise': compromise,
            'discovery': discovery,
            'method': method,
            'model': bundle.model_hash,
        },
        sort_keys=True,
    )
    ready = all(len(profile) >= 3 for profile in profiles)
    if not ready:
        st.info('Add three ratings per viewer, or try the editable example ratings above.')
    if st.button(
        'Find our movies' if members > 1 else 'Find my movies', type='primary', disabled=not ready
    ):
        with st.spinner('Learning your preferences and building the shortlist…'):
            if method == 'Personalized':
                predictions = np.stack([fit_viewer(bundle.model, profile) for profile in profiles])
            else:
                predictions = np.tile(catalog.popular_score.to_numpy(), (members, 1))
            seen = set().union(*(set(profile) for profile in profiles))
            result = rank_movies(
                catalog,
                predictions,
                seen,
                compromise=compromise,
                discovery=discovery,
                genres=tuple(genres),
                years=years,
                minimum_ratings=minimum,
            )
            st.session_state['recommendations'] = (signature, result)

    stored = st.session_state.get('recommendations')
    if stored is not None and stored[0] == signature:
        result = stored[1]
        st.subheader('Your movie-night shortlist' if members > 1 else 'Your shortlist')
        if result.empty:
            st.info(
                'No movies match these filters. Broaden the genres, years, or rating threshold.'
            )
        else:
            for position, row in result.iterrows():
                with st.container(border=True):
                    st.subheader(f'{position + 1}. {row.title}')
                    st.caption(row.genres.replace('|', ' · '))
                    labels = ' · '.join(
                        f'Viewer {member + 1}: {row[f"viewer_{member + 1}"]:.1f}'
                        for member in range(members)
                    )
                    st.write(f'Estimated ratings / 5 — {labels}')
                    st.caption(
                        f'Average: {row.predicted_average:.1f} · '
                        f'Lowest: {row.predicted_lowest:.1f} · '
                        f'{int(row["count"])} training ratings'
                    )
            st.caption(
                'Predictions are estimates, not guarantees. Anything any viewer rated is excluded.'
            )
    elif stored is not None:
        st.info('Preferences or filters changed. Click the recommendation button to refresh.')

    with st.expander('How the model works and measured results'):
        st.write(
            'Training learns user and movie embeddings from explicit ratings. We compare '
            'biased matrix factorization with a small neural interaction model, selecting '
            'the learned scorer with lower validation RMSE. New viewers get temporary '
            'embeddings fitted to their ratings while shared movie weights stay frozen.'
        )
        report_path = ROOT / 'artifacts/evaluation.json'
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if report['model_sha256'] == bundle.model_hash:
                st.write(f'Selected model: **{report["selected_model"]}**')
                st.dataframe(pd.DataFrame(report['rating_metrics']).T, width='stretch')
                st.dataframe(pd.DataFrame(report['ranking_metrics']).T, width='stretch')
                st.caption(
                    'Selection minimizes validation rating error. Lower rating error does '
                    'not necessarily mean better top-ten recommendations; compare both tables.'
                )
                st.caption(report['protocol'])
            else:
                st.info('Evaluation is stale. Run watch-together evaluate.')
        st.caption(
            'Offline metrics cover existing users, not unseen-viewer adaptation or actual '
            'group satisfaction. Genre/year filters and discovery are ranking controls. '
            'Ratings entered here remain in memory and are not saved to disk.'
        )


if __name__ == '__main__':
    main()
