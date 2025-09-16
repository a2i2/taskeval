# Contribution Guidelines

Thank you for contributing!

## Submitting bug reports and feature requests

Please feel free to submit bug reports and feature requests via the Issues tab. When doing so, please make sure you fill in all required information when prompted. Bug reports should have a clear and concise description, along with steps to reproduce the bug. Feature requests should contain enough context to be understood and actioned by maintainers.

## Submitting pull requests

* Please feel free to submit pull requests. When doing so, please make sure you fill in all required information when prompted.
* Before writing code, please create an issue outlining the design of the proposed feature.
* Ensure your code is consistent with the code style in this repository. Where possible, we will provide linter/formatter settings that will enable you to check your code quickly and easily.

### First-time setup

* Download and install the [latest version of git](https://git-scm.com/downloads).
* Configure git with your [username](https://help.github.com/articles/setting-your-username-in-git/) and [email](https://help.github.com/articles/setting-your-commit-email-address-in-git/):

```
git config --global user.name 'your name'
git config --global user.email 'your email'
```

* Make sure you have a [GitHub account](https://github.com/).
* Fork this repository to your GitHub account by clicking the [Fork](https://github.com/dstil/{this-repository}/fork) button.
* [Clone](https://help.github.com/articles/fork-a-repo/#step-2-create-a-local-clone-of-your-fork) your GitHub fork locally:

```
git clone https://github.com/{username}/{this-repository}
cd {this-repository}
```

* Add the main repository as a remote to update later:

```
git remote add upstream https://github.com/a2i2/{this-repository}
git fetch upstream
```

### Start coding

* Create a branch and identify the issue you would like to work on.
* Using your favourite editor, make your changes, [committing as you go](https://dont-be-afraid-to-commit.readthedocs.io/en/latest/git/commandlinegit.html#commit-your-changes).
* Include tests that cover any code changes you make. Make sure the test fails without your patch. [Run the tests](#running-the-tests).
* Push your commits to your fork and [create a pull request](https://help.github.com/articles/creating-a-pull-request/).

## Code of Conduct

Everyone contributing to this project and issue tracker is expected to follow the [Contributor Covenant Code of Conduct](https://contributor-covenant.org).
