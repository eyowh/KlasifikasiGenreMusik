---
name: Dungeon Studio
colors:
  surface: '#f9f9ff'
  surface-dim: '#cfdaf2'
  surface-bright: '#f9f9ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f0f3ff'
  surface-container: '#e7eeff'
  surface-container-high: '#dee8ff'
  surface-container-highest: '#d8e3fb'
  on-surface: '#111c2d'
  on-surface-variant: '#464555'
  inverse-surface: '#263143'
  inverse-on-surface: '#ecf1ff'
  outline: '#777587'
  outline-variant: '#c7c4d8'
  surface-tint: '#4d44e3'
  primary: '#3525cd'
  on-primary: '#ffffff'
  primary-container: '#4f46e5'
  on-primary-container: '#dad7ff'
  inverse-primary: '#c3c0ff'
  secondary: '#00687a'
  on-secondary: '#ffffff'
  secondary-container: '#57dffe'
  on-secondary-container: '#006172'
  tertiary: '#46494b'
  on-tertiary: '#ffffff'
  tertiary-container: '#5e6163'
  on-tertiary-container: '#dadcde'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#e2dfff'
  primary-fixed-dim: '#c3c0ff'
  on-primary-fixed: '#0f0069'
  on-primary-fixed-variant: '#3323cc'
  secondary-fixed: '#acedff'
  secondary-fixed-dim: '#4cd7f6'
  on-secondary-fixed: '#001f26'
  on-secondary-fixed-variant: '#004e5c'
  tertiary-fixed: '#e0e3e5'
  tertiary-fixed-dim: '#c4c7c9'
  on-tertiary-fixed: '#191c1e'
  on-tertiary-fixed-variant: '#444749'
  background: '#f9f9ff'
  on-background: '#111c2d'
  surface-variant: '#d8e3fb'
typography:
  display-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 48px
    fontWeight: '800'
    lineHeight: '1.1'
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 32px
    fontWeight: '700'
    lineHeight: '1.2'
    letterSpacing: -0.01em
  headline-lg-mobile:
    fontFamily: Plus Jakarta Sans
    fontSize: 28px
    fontWeight: '700'
    lineHeight: '1.2'
  headline-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 24px
    fontWeight: '600'
    lineHeight: '1.3'
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: '1.6'
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: '1.5'
  label-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '600'
    lineHeight: '1.4'
    letterSpacing: 0.05em
  label-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '500'
    lineHeight: '1.4'
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  base: 4px
  xs: 0.5rem
  sm: 1rem
  md: 1.5rem
  lg: 2.5rem
  xl: 4rem
  container-max: 1280px
  gutter: 24px
---

## Brand & Style
The design system for this platform focuses on an **Elegant High-Tech** aesthetic, balancing the precision of AI data analysis with the creative energy of music. The brand personality is professional yet vibrant, evoking a sense of discovery and structural clarity. 

The visual style follows a **Modern Corporate** approach with **Glassmorphism** highlights. It utilizes a sophisticated off-white environment to ensure high readability, while leveraging vibrant gradients and soft depth to indicate its advanced technological core. The interface should feel "airy" and expensive, prioritizing generous whitespace to prevent data-heavy views from feeling cluttered.

## Colors
The palette is anchored by a deep **Indigo (Primary)** which represents the "Dungeon" — a space for focused creation and deep learning. A **Vibrant Cyan (Accent)** is used sparingly for high-action items and progress indicators, providing a futuristic "glow" effect against the neutral backdrop.

The background is not pure white but a curated **Slate-50 (#F8FAFC)** to reduce eye strain and allow white cards to "pop" with subtle shadows. Text follows a strict hierarchy using deep slates for maximum contrast without the harshness of pure black.

## Typography
This design system uses a dual-font strategy. **Plus Jakarta Sans** is employed for headings to provide a friendly, modern, and slightly rounded geometric feel. **Inter** is used for all body text and UI labels to ensure maximum legibility at small sizes, maintaining a systematic and utilitarian feel for data classification.

Headline weights should remain Bold (700+) to create a strong visual anchor against the light background. Body text should maintain a 1.5x to 1.6x line height to support the "airy" brand narrative.

## Layout & Spacing
The system utilizes a **12-column fluid grid** for desktop and a **4-column grid** for mobile. A strict 8px/4px rhythm governs all spatial relationships. 

- **Desktop (1280px+):** 24px gutters, 80px side margins.
- **Tablet (768px-1279px):** 20px gutters, 40px side margins.
- **Mobile (Up to 767px):** 16px gutters, 20px side margins.

Content is grouped into "Data Clusters" using logical sections separated by `xl` (64px) spacing to emphasize the data-driven nature of the platform.

## Elevation & Depth
Depth is created through **Ambient Shadows** and **Tonal Layering** rather than heavy borders. The system uses three primary elevation levels:

1.  **Level 0 (Base):** The off-white background (#F8FAFC).
2.  **Level 1 (Surface):** Pure white (#FFFFFF) cards with a very soft, diffused shadow (0px 4px 20px rgba(0, 0, 0, 0.03)).
3.  **Level 2 (Active/Floating):** Used for modals and dropdowns, featuring a more pronounced shadow (0px 12px 32px rgba(79, 70, 229, 0.08)) with a slight Indigo tint to the shadow color.

Glassmorphism is applied to navigation bars and sidebars using a `backdrop-filter: blur(12px)` and a thin 1px white border at 40% opacity to simulate premium transparency.

## Shapes
In alignment with the "2xl" requirement, this design system utilizes a generous corner radius strategy. This softens the "high-tech" edge, making the AI feel more accessible and human-centric.

- **Standard UI elements (Inputs, Small Buttons):** 0.5rem (8px).
- **Cards and Containers (rounded-lg):** 1rem (16px).
- **Feature Blocks and Modal Containers (rounded-xl):** 1.5rem (24px).
- **Genre Badges:** Pill-shaped (full radius).

## Components

### Buttons
Primary buttons use a solid Indigo fill with white text. Hover states shift to a slightly darker Indigo with a soft Cyan outer glow. Tertiary buttons are transparent with Indigo text and a subtle background ghosting on hover.

### Cards
Cards are the core of the classification UI. They must have a pure white background, `rounded-xl` corners, and a 1px border (#E2E8F0) to ensure they stand out against the Slate-50 background. 

### Chips / Genre Tags
Each music genre should be represented by a "Pill" badge. These use high-contrast text on a low-opacity background of the same color (e.g., Synthwave uses a Cyan-100 background with Cyan-900 text).

### Input Fields
Inputs should be clean and minimalist. Use a 1px border (#CBD5E1) that transitions to a 2px Primary Indigo border on focus. Icons for "Search" or "Upload" should be thin-stroke (2pt) to maintain the elegant aesthetic.

### Visualization Elements
Data visualizations (genre probability charts, waveform analyzers) should use the Primary-to-Accent gradient (Indigo to Cyan) to represent AI confidence levels.