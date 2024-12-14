\ProvidesClass{clbeamer2024}

\LoadClass{beamer}
\usepackage{graphicx}
\usepackage{tikz}

% Couleurs personnalisées de l'ECL
\definecolor{ecltitre}{RGB}{33,78,178}      % A bold, deep blue
\definecolor{eclsoustitre}{RGB}{46,100,150} % A slightly lighter, muted blue

% Thème ECL
\setbeamercolor{title}{fg=ecltitre}
\setbeamercolor{frametitle}{fg=ecltitre}
\setbeamercolor{structure}{fg=eclsoustitre}

% Fond pour la première diapositive
\setbeamertemplate{background}{
    \ifnum\insertframenumber=0
        \begin{tikzpicture}[remember picture,overlay]
            \node[anchor=east, xshift=-2cm, yshift=1.4cm] at (current page.south) {
            };
        \end{tikzpicture}
    \fi
}

% Logo en haut à gauche pour la première diapositive
\titlegraphic{
    \ifnum\insertframenumber=0
        \begin{tikzpicture}[remember picture,overlay]
            \node[anchor=north west, xshift=0.5cm, yshift=-0.5cm] at (current page.north west) {
                \includegraphics[width=2cm]{logos/logo-pp.pdf}
            };
        \end{tikzpicture}
    \fi
}

% Footer avec blason et numéro de page à partir de la deuxième diapositive
\setbeamertemplate{footline}{
    \ifnum\insertframenumber>0
    \begin{tikzpicture}[remember picture,overlay]
        \node at ([yshift=0.7cm, xshift=0.7cm]current page.south west) {
            \includegraphics[height=0.8cm]{logos/blue.png}
        };
        \node at ([yshift=0.7cm, xshift=0.7cm]current page.south west) {
            \textbf{\textcolor{white}{\insertframenumber}}
        };
    \end{tikzpicture}
    \fi
}

\endinput
