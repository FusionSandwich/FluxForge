# Covariance and correlation inspection

Open **Tools → Covariance and Correlation** in the modern GUI. Load a JSON
artifact containing exactly one of `covariance`, `measurement_covariance` or
`rate_covariance`. Switch between the original covariance and its normalized
correlation, inspect exact values in the table, or export a PNG heatmap.

For example:

```json
{
  "title": "Activity covariance (Bq squared)",
  "labels": ["Co60", "Sc46", "Fixed input"],
  "covariance": [[4, -3, 0], [-3, 9, 0], [0, 0, 0]]
}
```

The correlation of the first two activities is -0.5. Correlation with the
zero-variance input is undefined, displayed as **N/A** and gray. Its original
covariance remains zero. Missing covariance is unavailable; no matrix is
invented from scalar error bars.

Labels may also be supplied as `observation_labels`, `parameter_names`,
`covariance_parameters` or `measurement_labels`. Rate artifacts can supply
`rates` rows containing `observation_id` or `reaction_id`. The label count must
match the matrix dimension. Include the input units in the title; covariance
entries have the product of their row and column units, whereas correlation is
dimensionless.

The viewer rejects nonfinite, nonsquare, asymmetric or indefinite matrices.
It supports singular positive-semidefinite matrices and preserves the supplied
covariance without diagonalization or regularization. Validation takes place
in standardized coordinates so changing units cannot hide invalid covariance.
Displaying an artifact does not qualify its source or uncertainty model.
