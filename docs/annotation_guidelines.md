# Annotation guidelines: is the product description supported by the MR?

For each row, read the attribute list (`mr`) and the product description (`ref`).
Decide whether the facts in the description are backed by the attributes.
Write exactly one label in `supported_by_MR`, in lower case: `yes`, `partial` or `no`.
Write a short reason in `notes` (for example "brand supported; 'black' not in MR").

## Labels

**yes** — The description says nothing factual that is missing from the MR.
Pure opinion or marketing words ("stylish", "elegant", "perfect for you") do not count as facts.

**partial** — The description is about the right product (at least one MR value, such as the
product type or the brand, is mentioned), BUT it also states at least one concrete fact
that the MR does not contain: a colour, material, size, quantity, carat weight, ingredient,
place of origin, feature, and so on.

**no** — Nothing product-specific in the description is backed by the MR: generic sales text,
brand history, shipping/gift messages, or a description of a different product.

## Decision steps
1. Does the description mention the product (type, name or brand from the MR)? If not → **no**.
2. Does it state any concrete fact (colour, material, number, feature…) not in the MR? If yes → **partial**.
3. Otherwise → **yes**.

## Examples
| Description | MR includes | Label | Why |
|---|---|---|---|
| "Get trendy with this printed top from Vero Moda. This black top will make you look elegant." | Vero Moda, top, printed (no colour) | partial | "black" is not in the MR |
| "This Jeweltouch necklace features complete craftsmanship, framed and mounted in an alloy. This necklace is unique among jewelry accessories." | Jewel Touch, necklace, alloy (in the name) | yes | every fact is in the MR; the rest is opinion |
| "The countdown is on! Time to select gifts for friends and families." | a shirt | no | nothing about the product |

Label each row on your own, without looking at anyone else's labels.
