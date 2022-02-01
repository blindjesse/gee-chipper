from functools import partial

import ee
import tensorflow as tf

from dataset import authenticate, get_asset_info, get_points, get_chips


session = authenticate()
dem_id = "CGIAR/SRTM90_V4"
slope = ee.Terrain.slope(ee.Image(dem_id))
dem_info = get_asset_info(dem_id, session)
scale = dem_info["bands"][0]["grid"]["affineTransform"]["scaleX"]


def get_loaded_chips(pt_tf):
    return tf.py_function(
        partial(
            get_chips,
            feature_image=dem_id,
            feature_bands=["elevation"],
            label_image=slope,
            label_bands=["slope"],
            scale=scale,
            session=session,
        ),
        [pt_tf],
        [tf.float32],
    )


points = get_points(10)
dataset = tf.data.Dataset.from_tensor_slices(points).map(get_loaded_chips)
