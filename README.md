\section*{Repository Overview}

\textbf{Title:} \textit{Testing a Siamese Interference Dark Sector with Triple Cosmological Falsification} \\
\textbf{Authors:} Cosmic Thinker \& ChatGPT ("Toko") \\
\textbf{Year:} 2025 \\
\textbf{DOI:} \url{https://doi.org/10.5281/zenodo.17685410}

\vspace{6pt}

This repository contains all code, data, figures and manuscript files associated with the preprint
\textit{``Testing a Siamese Interference Dark Sector with Triple Cosmological Falsification''}. The project presents the first full theoretical study of \textbf{Siamese Interference Cosmology}, an alternative to $\Lambda$CDM in which Dark Energy emerges as an interference effect between two CPT-symmetric branches of the Universe.

The model is implemented through a PNGB scalar field and evaluated using a strictly Popperian falsification approach: no late-time parameters are fitted to cosmological datasets. Instead, the model is tested through three independent cosmological falsifications:

\begin{itemize}
    \item \textbf{Expansion History ($H(z)$)} — compared against BAO + Cosmic Chronometers. \\
    \textit{Result:} the tested realisation over-accelerates the Universe.
    \item \textbf{Effective Equation of State ($w_{\mathrm{eff}}(z)$)} — derived directly from the dynamical evolution. \\
    \textit{Result:} $w_{\mathrm{eff}}(0) \approx -0.7$ in tension with the observational constraint $w \approx -1$.
    \item \textbf{Siamese Interference Entropy} ($S_{\mathrm{local}}$ and $dS_{\mathrm{local}}/dN$) — predicts the epoch of maximum anisotropy. \\
    \textit{Result:} open prediction, testable with FRBs, QSOs, and CMB EB/TB.
\end{itemize}

The objective of this work is not to confirm the model but to document transparently whether its predictions succeed or fail. Even though this particular realisation is disfavoured by current observations, it demonstrates a reproducible method for testing speculative dark sector physics.

A promising theoretical avenue is highlighted: if the interference becomes \textit{destructive} rather than constructive, the same framework could reproduce recent observational hints of a non-accelerating or decelerating Universe, as discussed by Lee et al. (2025). This direction is reserved for a future follow-up study.

\vspace{8pt}

\subsection*{Repository Contents}

\begin{itemize}
    \item \texttt{src/} — Python scripts to generate the three scientific figures.
    \item \texttt{results/} — numerical CSV dataset and produced figures.
    \item \texttt{paper/} — full \LaTeX{} source and compiled PDF.
\end{itemize}

All components required to reproduce the figures and results are included.

\vspace{8pt}

\subsection*{Citation}

If you use this work, please cite:

\begin{verbatim}
Cosmic Thinker & ChatGPT ("Toko"). (2025).
Testing a Siamese Interference Dark Sector with Triple Cosmological Falsification.
Zenodo. https://doi.org/10.5281/zenodo.17685410
\end{verbatim}

\vspace{6pt}

\subsection*{License}
Released under the \textbf{MIT License}.
