# FlyWire v783 data provenance

This record pins the two external inputs used by `flywire_gain_routing.py` and
`flywire_gain_routing_sensitivity.py`. The manuscript repository does not
duplicate these public data files.

## Connectivity table

- Local name: `Connectivity_783.parquet`
- Public source: Shiu et al., `Drosophila_brain_model`
- Pinned commit: `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960`
- URL: `https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960/Connectivity_783.parquet`
- SHA-256: `efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347`
- Analysis rows after loading: 15,091,983 directed neuron-pair edges.

## Complete v783 node universe

- Public source: Codex FAFB v783 `neurons.csv.gz`
- URL: `https://storage.googleapis.com/flywire-data/codex/data/fafb/783/neurons.csv.gz`
- Download SHA-256: `6a6b3759e635f0f35a677d169052362131ec61d95f55919298b55c43fce4e719`
- Transformation: retain the public row order and `root_id`; write the exact
  two-column file `,Completed`, with `True` for every row.
- Derived local name: `Completeness_783_all_139255.csv`
- Derived SHA-256: `d144a7a4d5b25c13d452c23ad04bf90ccbbded2c03b492b8eb024d13ac6a987d`
- Rows: 139,255 unique published neurons.

The Shiu connectivity table contains 138,639 of these roots. The remaining 616
published neurons have no row in that model graph and are retained as zero-edge
nodes. This preserves the complete published node universe without inventing
connections.

## Exact preparation

From `energy_nonnormal/`, run:

```bash
python3 validation/prepare_flywire_v783_inputs.py ../flywire_spontaneous
```

The helper downloads both pinned sources, verifies their hashes, materializes
the complete two-column node table, and refuses silent source drift. Existing
files are only accepted when their SHA-256 values match.

Task endpoint IDs are copied from the Shiu et al. reproduction notebook at the
same commit. Three v630 roots changed before v783; their new roots and stable
supervoxels are recorded in `flywire_gain_routing_results.json`. That JSON also
records every selected gain-module ID, numerical parameter, null draw, and the
hashes above.

This is a connectome-constrained in-silico intervention. It does not claim that
the imposed gain fields were observed in vivo.
