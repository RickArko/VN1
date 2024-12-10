import polars as pl


def load_full_data() -> pl.DataFrame:
    """Read the full dataset from the datasource.ai.

    Returns:
        pl.DataFrame: Combined phase1 and phase2 data.
    """
    phase1_url = 'https://www.datasource.ai/attachments/eyJpZCI6Ijk4NDYxNjE2NmZmZjM0MGRmNmE4MTczOGMyMzI2ZWI2LmNzdiIsInN0b3JhZ2UiOiJzdG9yZSIsIm1ldGFkYXRhIjp7ImZpbGVuYW1lIjoiUGhhc2UgMCAtIFNhbGVzLmNzdiIsInNpemUiOjEwODA0NjU0LCJtaW1lX3R5cGUiOiJ0ZXh0L2NzdiJ9fQ'
    phase1 = pl.read_csv(phase1_url).sort('Client', 'Warehouse', 'Product')
    phase2_url = 'https://www.datasource.ai/attachments/eyJpZCI6ImM2OGQxNGNmNTJkZDQ1MTUyZTg0M2FkMDAyMjVlN2NlLmNzdiIsInN0b3JhZ2UiOiJzdG9yZSIsIm1ldGFkYXRhIjp7ImZpbGVuYW1lIjoiUGhhc2UgMSAtIFNhbGVzLmNzdiIsInNpemUiOjEwMTgzOTYsIm1pbWVfdHlwZSI6InRleHQvY3N2In19'
    phase2 = pl.read_csv(phase2_url).sort('Client', 'Warehouse', 'Product').drop('Client', 'Warehouse', 'Product')
    wide = pl.concat([phase1, phase2], how='horizontal')
    return wide


def process_wide_df(wide: pl.DataFrame) -> pl.DataFrame:
    """Convert wide format to long format via unpivot.

    Args:
        wide (pl.DataFrame): index ids, date columns, and y values.

    Returns:
        pl.DataFrame: unique_id, ds, and y columns.
    """
    long = wide.unpivot(index=['Client', 'Warehouse', 'Product'],
                        variable_name='ds',
                        value_name='y')
    long = long.with_columns(
        unique_id=pl.col('Client').cast(pl.Utf8) + '-' + pl.col('Warehouse').cast(pl.Utf8) + '-' + pl.col('Product').cast(pl.Utf8),
        ds=pl.col('ds').cast(pl.Date)
    )
    return long
