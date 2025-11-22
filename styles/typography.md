# Typography Decisions

- Primary font: Inter (fallbacks: "Segoe UI", system-ui, sans-serif).
- Secondary font (monospace/code): "IBM Plex Mono", "SFMono-Regular", Menlo, monospace.
- Base figure/background choice: light backgrounds with dark text for print clarity; dark-neutral background is reserved for hero slides if needed.
- Heading hierarchy for slides/notebooks:
  - H1: 32-36 pt, bold, color `primary.700`
  - H2: 26-30 pt, semibold, color `primary.600`
  - H3: 20-22 pt, semibold, color `primary.500`
- Figure text sizing (matplotlib/plotly defaults):
  - Title: 16-18 pt, bold; Subtitle: 13-14 pt, regular
  - Axis labels: 12-13 pt; Tick labels: 11-12 pt
  - Legend: 11-12 pt; Annotation callouts: 11-12 pt with `primary.700`
- Line widths & grid: 1.8-2.2 pt lines; light gridlines using `primary.100` on light background.
- Callout colors: use `accent.warning` for cautions, `accent.danger` for risks or errors.
