# ParSEL project website

The paper and displayed project name remain **ParSEL**. The repository name and
canonical project-page URL are lowercase:
<https://bardofcodes.github.io/parsel/>.

This `gh-pages` branch contains the static research website and its original
figures, animations, and Blender demonstration. The `main` branch contains the
one-time cleaned code publication snapshot.

## GitHub Pages deployment

After renaming the repository to `parsel`, select **Deploy from a branch** in
Settings → Pages, with branch `gh-pages` and folder `/ (root)`.
Keep the older `gh_pages` branch until the new deployment is verified.

GitHub redirects renamed repository links, but not project-site URLs. Arrange a
redirect for the old `/parSEL/` page before changing the canonical URL; the
personal website and release-news update are separate, deferred tasks.

For a local preview, run `python -m http.server 8765 --bind 127.0.0.1` from this
branch and open <http://127.0.0.1:8765/>. No build step is required.

This website has been adapted from the [Nerfies website](https://nerfies.github.io) code.

## Template attribution

The original template is from the [Nerfies website](https://nerfies.github.io).

If you find Nerfies useful for your work please cite:
```
@article{park2021nerfies
  author    = {Park, Keunhong and Sinha, Utkarsh and Barron, Jonathan T. and Bouaziz, Sofien and Goldman, Dan B and Seitz, Steven M. and Martin-Brualla, Ricardo},
  title     = {Nerfies: Deformable Neural Radiance Fields},
  journal   = {ICCV},
  year      = {2021},
}
```

## Website license
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
