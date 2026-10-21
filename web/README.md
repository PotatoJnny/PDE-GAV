# PDE Agent panel

This is the fixed GPT-sign-in web panel for PDE Agent campaigns.

The customer experience has two primary surfaces:

- **Running now** shows the current scientific phase, campaign round, progress,
  intermediate solver errors, Generator learning curves, basis quality, and case mix.
- **Results** shows the final hidden-test error, round-by-round solver performance,
  case-level error distributions, physics checks, Generator quality, memory summaries,
  and the final solver action.

The page consumes only the product-facing `pde_panel_snapshot_v1` model. Cluster
paths, scheduler ids, commands, and raw logs are not part of that model.

## Development

```bash
npm ci
npm run dev
npm test
```

The live deployment stores the latest metric-only outbound snapshot from the compute cluster
and refreshes the customer view every five seconds. Reads are link-accessible;
campaign metrics require GPT sign-in, and writes use a separate synchronization credential.
