# Contributing Guide

This repository follows a simple but professional workflow to keep research code reliable and easy to review.

## 1) Branch Strategy

- `main`: stable branch for validated work.
- Feature branches for all changes:
  - `feat/phase4-backtest-engine`
  - `fix/preprocessing-date-index`
  - `docs/readme-update`
  - `chore/repo-maintenance`

## 2) Commit Message Standard (Conventional Commits)

Use this format:

`<type>(<scope>): <short summary>`

Examples:

- `feat(backtest): add rolling monthly walk-forward engine`
- `fix(optimizer): enforce crypto-cap constraint check`
- `docs(todo): update roadmap for phase 4`
- `chore(repo): initialize git and baseline project files`
- `test(metrics): add max drawdown unit tests`

### Recommended types

- `feat`: new functionality
- `fix`: bug fix
- `docs`: documentation only
- `test`: tests added/updated
- `refactor`: code cleanup without behavior change
- `chore`: maintenance / tooling / repo setup

## 3) Commit Size Rules

- Keep each commit focused on one logical change.
- Prefer small commits over large mixed commits.
- Do not mix docs, refactor, and feature behavior in one commit when avoidable.

## 4) Suggested Scope Names

- `config`
- `loader`
- `preprocessing`
- `metrics`
- `benchmarks`
- `optimizer`
- `backtest`
- `notebooks`
- `todo`
- `repo`

## 5) Before Every Commit

- Run the relevant script(s) end-to-end when possible.
- Confirm no accidental files are staged (especially large data files).
- Ensure TODO and README stay aligned with implementation status.

## 6) Tagging Milestones

Suggested lightweight tags:

- `v0.1-foundation` (setup + data pipeline)
- `v0.2-benchmarks` (metrics + benchmarks)
- `v0.3-static-minvar` (first optimizer, static)
- `v0.4-walkforward` (rolling out-of-sample backtest)

## 7) GitHub Pull Request Title Pattern

`<type>(<scope>): <summary>`

Keep the PR description short and include:

- What changed
- Why it changed
- How to run/validate
- Known limitations
