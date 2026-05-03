# Project Context

## Why This Exists

This project is an attempt to convert Amrut's discretionary long-duration "coil effect" trading knowledge into a repeatable system that can be scaled, tested, and eventually used by others.

The broader intent is to build something that outlasts any single individual. The goal is not just to assist Amrut in his own workflow, but to encode the strategy in a durable research and screening system.

## People Involved

### Amrut

- Has more than 18 years of trading experience
- Is also a CA / CS
- Has traded international markets successfully
- Has made profits using this strategy in practice
- Provides the market knowledge, pattern intuition, and real-world validation

### Ankit Mishra

- Working together on this effort with Amrut
- Part of the core build / research loop

### Anubhav + Technical Build Side

- Bringing strong technical depth in computers and technology over roughly 20 years
- Bringing roughly 5+ years of experience in ML, time series, and applied systems work
- Focused on turning discretionary pattern knowledge into a reproducible technical system
- Motivated by solving hard problems and building durable tools

## Current Working Thesis

The strategy should be built in two layers:

1. A quantitative monthly-data screen that reduces the stock universe to candidates likely to contain long-duration coil structures.
2. A visual analysis layer that evaluates the chart geometry more deeply, because the final pattern judgment is partly geometric and cannot be captured cleanly by raw numeric filters alone.

## Current Status

- A first-pass monthly screener has been implemented.
- Initial results show the pipeline works, but the screener still over-ranks some strong secular uptrends that are near highs without being true coil structures.
- The next phase is refinement of the screen, followed by broader testing and visual standardization.

## Immediate Direction

- Refine the numeric screener to better separate true long-duration compression from simple strong uptrends.
- Expand testing on a broader universe.
- Use Amrut's feedback to validate whether top-ranked names actually fit the strategy.
- Build toward a labeled visual dataset and eventual ML / CV ranking layer.
