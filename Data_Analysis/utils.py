import random

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

# Set the seaborn style
sns.set(style="whitegrid")

MET_PALETTES = dict(
    Archambault=dict(
        colors=(
            "#88a0dc",
            "#381a61",
            "#7c4b73",
            "#ed968c",
            "#ab3329",
            "#e78429",
            "#f9d14a",
        ),
        order=(2, 7, 5, 1, 6, 4, 3),
        colorblind=True,
    ),
    Austria=dict(
        colors=(
            "#a40000",
            "#16317d",
            "#007e2f",
            "#ffcd12",
            "#b86092",
            "#721b3e",
            "#00b7a7",
        ),
        order=(1, 2, 3, 4, 6, 5, 7),
        colorblind=False,
    ),
    Benedictus=dict(
        colors=(
            "#9a133d",
            "#b93961",
            "#d8527c",
            "#f28aaa",
            "#f9b4c9",
            "#f9e0e8",
            "#ffffff",
            "#eaf3ff",
            "#c5daf6",
            "#a1c2ed",
            "#6996e3",
            "#4060c8",
            "#1a318b",
        ),
        order=(9, 5, 11, 1, 7, 3, 13, 4, 8, 2, 12, 6, 10),
        colorblind=False,
    ),
    Cassatt1=dict(
        colors=(
            "#b1615c",
            "#d88782",
            "#e3aba7",
            "#edd7d9",
            "#c9c9dd",
            "#9d9dc7",
            "#8282aa",
            "#5a5a83",
        ),
        order=(3, 6, 1, 8, 4, 5, 2, 7),
        colorblind=True,
    ),
    Cassatt2=dict(
        colors=(
            "#2d223c",
            "#574571",
            "#90719f",
            "#b695bc",
            "#dec5da",
            "#c1d1aa",
            "#7fa074",
            "#466c4b",
            "#2c4b27",
            "#0e2810",
        ),
        order=(7, 3, 9, 1, 5, 6, 2, 10, 4, 8),
        colorblind=True,
    ),
    Cross=dict(
        colors=(
            "#c969a1",
            "#ce4441",
            "#ee8577",
            "#eb7926",
            "#ffbb44",
            "#859b6c",
            "#62929a",
            "#004f63",
            "#122451",
        ),
        order=(4, 7, 1, 8, 2, 6, 3, 5, 9),
        colorblind=False,
    ),
    Degas=dict(
        colors=(
            "#591d06",
            "#96410e",
            "#e5a335",
            "#556219",
            "#418979",
            "#2b614e",
            "#053c29",
        ),
        order=(5, 2, 1, 3, 4, 7, 6),
        colorblind=False,
    ),
    Demuth=dict(
        colors=(
            "#591c19",
            "#9b332b",
            "#b64f32",
            "#d39a2d",
            "#f7c267",
            "#b9b9b8",
            "#8b8b99",
            "#5d6174",
            "#41485f",
            "#262d42",
        ),
        order=(9, 5, 1, 7, 3, 4, 8, 2, 6, 10),
        colorblind=True,
    ),
    Derain=dict(
        colors=(
            "#efc86e",
            "#97c684",
            "#6f9969",
            "#aab5d5",
            "#808fe1",
            "#5c66a8",
            "#454a74",
        ),
        order=(4, 2, 5, 7, 1, 3, 6),
        colorblind=True,
    ),
    Egypt=dict(
        colors=("#dd5129", "#0f7ba2", "#43b284", "#fab255"),
        order=(1, 2, 3, 4),
        colorblind=True,
    ),
    Gauguin=dict(
        colors=("#b04948", "#811e18", "#9e4013", "#c88a2c", "#4c6216", "#1a472a"),
        order=(2, 5, 4, 3, 1, 6),
        colorblind=False,
    ),
    Greek=dict(
        colors=("#3c0d03", "#8d1c06", "#e67424", "#ed9b49", "#f5c34d"),
        order=(2, 3, 5, 1, 4),
        colorblind=True,
    ),
    Hiroshige=dict(
        colors=(
            "#e76254",
            "#ef8a47",
            "#f7aa58",
            "#ffd06f",
            "#ffe6b7",
            "#aadce0",
            "#72bcd5",
            "#528fad",
            "#376795",
            "#1e466e",
        ),
        order=(6, 2, 9, 3, 7, 5, 1, 10, 4, 8),
        colorblind=True,
    ),
    Hokusai1=dict(
        colors=(
            "#6d2f20",
            "#b75347",
            "#df7e66",
            "#e09351",
            "#edc775",
            "#94b594",
            "#224b5e",
        ),
        order=(2, 7, 4, 6, 5, 1, 3),
        colorblind=False,
    ),
    Hokusai2=dict(
        colors=("#abc9c8", "#72aeb6", "#4692b0", "#2f70a1", "#134b73", "#0a3351"),
        order=(5, 2, 4, 1, 6, 3),
        colorblind=True,
    ),
    Hokusai3=dict(
        colors=("#d8d97a", "#95c36e", "#74c8c3", "#5a97c1", "#295384", "#0a2e57"),
        order=(4, 2, 5, 3, 1, 6),
        colorblind=True,
    ),
    Homer1=dict(
        colors=(
            "#551f00",
            "#a62f00",
            "#df7700",
            "#f5b642",
            "#fff179",
            "#c3f4f6",
            "#6ad5e8",
            "#32b2da",
        ),
        order=(6, 3, 2, 7, 4, 8, 5, 1),
        colorblind=False,
    ),
    Homer2=dict(
        colors=(
            "#bf3626",
            "#e9724c",
            "#e9851d",
            "#f9c53b",
            "#aeac4c",
            "#788f33",
            "#165d43",
        ),
        order=(3, 7, 1, 4, 6, 2, 5),
        colorblind=False,
    ),
    Ingres=dict(
        colors=(
            "#041d2c",
            "#06314e",
            "#18527e",
            "#2e77ab",
            "#d1b252",
            "#a97f2f",
            "#7e5522",
            "#472c0b",
        ),
        order=(4, 5, 3, 6, 2, 7, 1, 8),
        colorblind=True,
    ),
    Isfahan1=dict(
        colors=(
            "#4e3910",
            "#845d29",
            "#d8c29d",
            "#4fb6ca",
            "#178f92",
            "#175f5d",
            "#1d1f54",
        ),
        order=(5, 2, 4, 6, 1, 7, 3),
        colorblind=True,
    ),
    Isfahan2=dict(
        colors=("#d7aca1", "#ddc000", "#79ad41", "#34b6c6", "#4063a3"),
        order=(4, 2, 3, 5, 1),
        colorblind=True,
    ),
    Java=dict(
        colors=("#663171", "#cf3a36", "#ea7428", "#e2998a", "#0c7156"),
        order=(1, 4, 2, 5, 3),
        colorblind=True,
    ),
    Johnson=dict(
        colors=("#a00e00", "#d04e00", "#f6c200", "#0086a8", "#132b69"),
        order=(3, 1, 4, 2, 5),
        colorblind=True,
    ),
    Juarez=dict(
        colors=("#a82203", "#208cc0", "#f1af3a", "#cf5e4e", "#637b31", "#003967"),
        order=(1, 2, 3, 4, 5, 6),
        colorblind=False,
    ),
    Kandinsky=dict(
        colors=("#3b7c70", "#ce9642", "#898e9f", "#3b3a3e"),
        order=(1, 2, 3, 4),
        colorblind=True,
    ),
    Klimt=dict(
        colors=("#df9ed4", "#c93f55", "#eacc62", "#469d76", "#3c4b99", "#924099"),
        order=(5, 2, 3, 4, 6, 1),
        colorblind=False,
    ),
    Lakota=dict(
        colors=("#04a3bd", "#f0be3d", "#931e18", "#da7901", "#247d3f", "#20235b"),
        order=(1, 2, 3, 4, 5, 6),
        colorblind=False,
    ),
    Manet=dict(
        colors=(
            "#3b2319",
            "#80521c",
            "#d29c44",
            "#ebc174",
            "#ede2cc",
            "#7ec5f4",
            "#4585b7",
            "#225e92",
            "#183571",
            "#43429b",
            "#5e65be",
        ),
        order=(8, 3, 10, 4, 7, 9, 11, 2, 6, 1, 5),
        colorblind=False,
    ),
    Monet=dict(
        colors=(
            "#4e6d58",
            "#749e89",
            "#abccbe",
            "#e3cacf",
            "#c399a2",
            "#9f6e71",
            "#41507b",
            "#7d87b2",
            "#c2cae3",
        ),
        order=(2, 5, 8, 3, 4, 9, 1, 6, 7),
        colorblind=False,
    ),
    Moreau=dict(
        colors=(
            "#421600",
            "#792504",
            "#bc7524",
            "#8dadca",
            "#527baa",
            "#104839",
            "#082844",
        ),
        order=(2, 5, 3, 4, 7, 1, 6),
        colorblind=False,
    ),
    Morgenstern=dict(
        colors=(
            "#7c668c",
            "#b08ba5",
            "#dfbbc8",
            "#ffc680",
            "#ffb178",
            "#db8872",
            "#a56457",
        ),
        order=(7, 5, 4, 6, 3, 2, 1),
        colorblind=True,
    ),
    Nattier=dict(
        colors=(
            "#52271c",
            "#944839",
            "#c08e39",
            "#7f793c",
            "#565c33",
            "#184948",
            "#022a2a",
        ),
        order=(1, 6, 3, 4, 7, 2, 5),
        colorblind=False,
    ),
    Navajo=dict(
        colors=("#660d20", "#e59a52", "#edce79", "#094568", "#e1c59a"),
        order=(1, 2, 3, 4, 5),
        colorblind=False,
    ),
    NewKingdom=dict(
        colors=("#e1846c", "#9eb4e0", "#e6bb9e", "#9c6849", "#735852"),
        order=(2, 1, 3, 4, 5),
        colorblind=False,
    ),
    Nizami=dict(
        colors=(
            "#dd7867",
            "#b83326",
            "#c8570d",
            "#edb144",
            "#8cc8bc",
            "#7da7ea",
            "#5773c0",
            "#1d4497",
        ),
        order=(5, 2, 6, 8, 3, 7, 4, 1),
        colorblind=False,
    ),
    OKeeffe1=dict(
        colors=(
            "#6b200c",
            "#973d21",
            "#da6c42",
            "#ee956a",
            "#fbc2a9",
            "#f6f2ee",
            "#bad6f9",
            "#7db0ea",
            "#447fdd",
            "#225bb2",
            "#133e7e",
        ),
        order=(8, 6, 1, 4, 10, 3, 11, 5, 2, 7, 9),
        colorblind=True,
    ),
    OKeeffe2=dict(
        colors=(
            "#fbe3c2",
            "#f2c88f",
            "#ecb27d",
            "#e69c6b",
            "#d37750",
            "#b9563f",
            "#92351e",
        ),
        order=(7, 1, 6, 4, 2, 5, 3),
        colorblind=True,
    ),
    Paquin=dict(
        colors=(
            "#831818",
            "#c62320",
            "#f05b43",
            "#f78462",
            "#feac81",
            "#f7dea3",
            "#ced1af",
            "#98ab76",
            "#748f46",
            "#47632a",
            "#275024",
        ),
        order=(10, 6, 1, 8, 4, 3, 5, 9, 2, 7, 11),
        colorblind=False,
    ),
    Peru1=dict(
        colors=("#b5361c", "#e35e28", "#1c9d7c", "#31c7ba", "#369cc9", "#3a507f"),
        order=(3, 1, 5, 2, 4, 6),
        colorblind=False,
    ),
    Peru2=dict(
        colors=(
            "#65150b",
            "#961f1f",
            "#c0431f",
            "#b36c06",
            "#f19425",
            "#c59349",
            "#533d14",
        ),
        order=(4, 1, 3, 5, 2, 7, 6),
        colorblind=False,
    ),
    Pillement=dict(
        colors=("#a9845b", "#697852", "#738e8e", "#44636f", "#2b4655", "#0f252f"),
        order=(4, 3, 2, 5, 1, 6),
        colorblind=True,
    ),
    Pissaro=dict(
        colors=(
            "#134130",
            "#4c825d",
            "#8cae9e",
            "#8dc7dc",
            "#508ca7",
            "#1a5270",
            "#0e2a4d",
        ),
        order=(6, 2, 4, 1, 7, 5, 3),
        colorblind=False,
    ),
    Redon=dict(
        colors=(
            "#5b859e",
            "#1e395f",
            "#75884b",
            "#1e5a46",
            "#df8d71",
            "#af4f2f",
            "#d48f90",
            "#732f30",
            "#ab84a5",
            "#59385c",
            "#d8b847",
            "#b38711",
        ),
        order=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        colorblind=False,
    ),
    Renoir=dict(
        colors=(
            "#17154f",
            "#2f357c",
            "#6c5d9e",
            "#9d9cd5",
            "#b0799a",
            "#f6b3b0",
            "#e48171",
            "#bf3729",
            "#e69b00",
            "#f5bb50",
            "#ada43b",
            "#355828",
        ),
        order=(2, 5, 9, 12, 3, 8, 7, 10, 4, 1, 6, 11),
        colorblind=False,
    ),
    Robert=dict(
        colors=("#11341a", "#375624", "#6ca4a0", "#487a7c", "#18505f", "#062e3d"),
        order=(2, 5, 3, 1, 6, 4),
        colorblind=False,
    ),
    Signac=dict(
        colors=(
            "#fbe183",
            "#f4c40f",
            "#fe9b00",
            "#d8443c",
            "#9b3441",
            "#de597c",
            "#e87b89",
            "#e6a2a6",
            "#aa7aa1",
            "#9f5691",
            "#633372",
            "#1f6e9c",
            "#2b9b81",
            "#92c051",
        ),
        order=(13, 3, 2, 1, 11, 5, 8, 14, 12, 10, 7, 4, 6, 9),
        colorblind=False,
    ),
    Stevens=dict(
        colors=("#042e4e", "#307d7f", "#598c4c", "#ba5c3f", "#a13213", "#470c00"),
        order=(4, 2, 3, 5, 1, 6),
        colorblind=False,
    ),
    Tam=dict(
        colors=(
            "#ffd353",
            "#ffb242",
            "#ef8737",
            "#de4f33",
            "#bb292c",
            "#9f2d55",
            "#62205f",
            "#341648",
        ),
        order=(3, 8, 1, 6, 2, 7, 4, 5),
        colorblind=True,
    ),
    Tara=dict(
        colors=("#eab1c6", "#d35e17", "#e18a1f", "#e9b109", "#829d44"),
        order=(1, 3, 2, 5, 4),
        colorblind=False,
    ),
    Thomas=dict(
        colors=(
            "#b24422",
            "#c44d76",
            "#4457a5",
            "#13315f",
            "#b1a1cc",
            "#59386c",
            "#447861",
            "#7caf5c",
        ),
        order=(3, 2, 8, 6, 1, 4, 7, 5),
        colorblind=False,
    ),
    Tiepolo=dict(
        colors=(
            "#802417",
            "#c06636",
            "#ce9344",
            "#e8b960",
            "#646e3b",
            "#2b5851",
            "#508ea2",
            "#17486f",
        ),
        order=(1, 2, 8, 4, 3, 5, 7, 6),
        colorblind=False,
    ),
    Troy=dict(
        colors=(
            "#421401",
            "#6c1d0e",
            "#8b3a2b",
            "#c27668",
            "#7ba0b4",
            "#44728c",
            "#235070",
            "#0a2d46",
        ),
        order=(2, 7, 4, 5, 1, 8, 3, 6),
        colorblind=True,
    ),
    Tsimshian=dict(
        colors=(
            "#582310",
            "#aa361d",
            "#82c45f",
            "#318f49",
            "#0cb4bb",
            "#2673a3",
            "#473d7d",
        ),
        order=(6, 1, 7, 4, 1, 5, 3),
        colorblind=False,
    ),
    VanGogh1=dict(
        colors=(
            "#2c2d54",
            "#434475",
            "#6b6ca3",
            "#969bc7",
            "#87bcbd",
            "#89ab7c",
            "#6f9954",
        ),
        order=(3, 5, 7, 4, 6, 2, 1),
        colorblind=False,
    ),
    VanGogh2=dict(
        colors=(
            "#bd3106",
            "#d9700e",
            "#e9a00e",
            "#eebe04",
            "#5b7314",
            "#c3d6ce",
            "#89a6bb",
            "#454b87",
        ),
        order=(1, 5, 8, 2, 7, 4, 6, 3),
        colorblind=False,
    ),
    VanGogh3=dict(
        colors=(
            "#e7e5cc",
            "#c2d6a4",
            "#9cc184",
            "#669d62",
            "#447243",
            "#1f5b25",
            "#1e3d14",
            "#192813",
        ),
        order=(7, 5, 1, 4, 8, 2, 3, 6),
        colorblind=True,
    ),
    Veronese=dict(
        colors=(
            "#67322e",
            "#99610a",
            "#c38f16",
            "#6e948c",
            "#2c6b67",
            "#175449",
            "#122c43",
        ),
        order=(5, 1, 7, 2, 3, 6, 4),
        colorblind=True,
    ),
    Wissing=dict(
        colors=("#4b1d0d", "#7c291e", "#ba7233", "#3a4421", "#2d5380"),
        order=(2, 3, 5, 4, 1),
        colorblind=False,
    ),
)

COLORBLIND_PALETTES_NAMES = (
    "Archambault",
    "Cassatt1",
    "Cassatt2",
    "Demuth",
    "Derain",
    "Egypt",
    "Greek",
    "Hiroshige",
    "Hokusai2",
    "Hokusai3",
    "Ingres",
    "Isfahan1",
    "Isfahan2",
    "Java",
    "Johnson",
    "Kandinsky",
    "Morgenstern",
    "OKeeffe1",
    "OKeeffe2",
    "Pillement",
    "Tam",
    "Troy",
    "VanGogh3",
    "Veronese",
)


def plot_histogram(
    dataframe, column, color=None, xlabel=None, ylabel=None, title=None, bins=10
):
    if color is None:
        palette = MET_PALETTES[random.choice(COLORBLIND_PALETTES_NAMES)]
        color = palette["colors"][random.choice(range(0, len(palette["colors"])))]

    plt.figure(figsize=(10, 6))
    plt.hist(dataframe[column], bins=bins, edgecolor="black", color=color)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)

    plt.show()


def plot_histogram_plotly(dataframe, column, title, x_label, y_label, bins, fig_name):
    # Colorblind-friendly color palette
    colors = MET_PALETTES["NewKingdom"]["colors"]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=dataframe[column],
            nbinsx=bins,
            marker_color=colors[0],  # Use only the first color
            marker=dict(line=dict(color="DarkSlateGrey", width=1)),  # Set edge color
        )
    )

    fig.update_traces(
        texttemplate="%{y}", textposition="auto"  # Display count in each bin
    )

    fig.update_layout(
        title_text=title,  # title of plot
        xaxis_title_text=x_label,  # x axis label
        yaxis_title_text=y_label,  # y axis label
        template="plotly_white",  # for a more professional look
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
    )

    fig.update_yaxes(automargin=True)  # Enable automatic margins for y-axis

    fig.show()
    if fig_name is not None:
        fig.write_image(fig_name)


def plot_pie_chart(dataframe, column, title=None, colors=sns.color_palette("pastel")):
    grouped_data = dataframe[column].value_counts()
    labels = grouped_data.index
    sizes = grouped_data.values
    colors = MET_PALETTES["Hokusai2"]["colors"]

    plt.figure(figsize=(10, 10))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)

    if title is not None:
        plt.title(title)

    plt.axis("equal")
    plt.show()


def plot_pie_chart_plotly(dataframe, column, title=None, fig_name=None):
    # Colorblind-friendly color palette
    colors = MET_PALETTES["NewKingdom"]["colors"]

    grouped_data = dataframe[column].value_counts()
    labels = grouped_data.index
    values = grouped_data.values

    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_traces(
        hoverinfo="label+percent",
        textinfo="percent",
        textfont_size=20,
        marker=dict(colors=colors, line=dict(color="rgba(255,255,255,0.2)", width=2)),
    )

    fig.update_layout(
        title_text=title,  # title of plot
        autosize=False,  # disable autosize to set custom width and height
        width=800,  # width of the plot
        height=600,  # height of the plot
        plot_bgcolor="rgba(0,0,0,0)",  # Transparent background
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
        template="plotly_white",  # for a more professional look
        margin=dict(l=50, r=50, b=50, t=50),  # Set custom margins
    )

    fig.show()
    if fig_name is not None:
        fig.write_image(fig_name)