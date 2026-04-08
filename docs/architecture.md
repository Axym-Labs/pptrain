# Notes

`pptrain` stays useful by keeping the upstream training path small:

1. build synthetic upstream data
2. train a causal LM on it
3. export a transfer bundle

Everything else should stay secondary to that path.
