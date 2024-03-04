
## Documentation

Documentation should be viewable here: [https://github.com/pages/EQuS/qcsys/](https://github.com/pages/EQuS/qcsys/) 

### Build and view locally

To view documentation locally, plesae make sure the install the requirements under the `docs` extra, as specified above. Then, run the following:

```
mkdocs serve
```

The documentation should now be at the url provided by the above command. 

### Updating Docs

The documentation should be updated automatically when any changes are made to the `main` branch. However, updates can also be forced by running:

```
mkdocs gh-deploy --force
```
This will build your documentation and deploy it to a branch gh-pages in your repository.