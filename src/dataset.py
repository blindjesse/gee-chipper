import io
import json
import os
from typing import List, Union

import ee
from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account
import numpy as np
import tensorflow as tf


def authenticate(key_file: str = os.environ["GA_AUTH_FILE"]) -> AuthorizedSession:

    gcs_credentials = service_account.Credentials.from_service_account_file(
        os.environ["GA_AUTH_FILE"]
    )
    ee_creds = ee.ServiceAccountCredentials(os.environ["GEE_SERVICE_ACCOUNT"], key_file)
    ee.Initialize(ee_creds)
    scoped_credentials = gcs_credentials.with_scopes(
        ["https://www.googleapis.com/auth/cloud-platform"]
    )

    return AuthorizedSession(scoped_credentials)


compute_url = "https://earthengine.googleapis.com/v1beta/projects/earthengine-public/image:computePixels"


def get_asset_url(asset_id):
    name = f"projects/earthengine-public/assets/{asset_id}"
    return f"https://earthengine.googleapis.com/v1beta/{name}"


def get_asset_info(asset_id, session):
    return json.loads(session.get(get_asset_url(asset_id)).content)


# def get_computed_chip(coords, image, band_ids, scale, session):

# def get_asset_chip(coords, asset_id, band_ids, scale, session):


def get_chip(
    coords: List,
    image: Union[str, ee.Image],
    band_ids: List,
    scale: float,
    session: AuthorizedSession,
):
    query = {
        "fileFormat": "NPY",
        #        "bandIds": band_ids,
        "grid": {
            "affineTransform": {
                "scaleX": scale,
                "scaleY": scale,
                "translateX": coords[0],
                "translateY": coords[1],
            },
            "dimensions": {"width": 512, "height": 512},
        },
    }

    if isinstance(image, (ee.Image)):
        url = compute_url
        query["expression"] = ee.serializer.encode(image)
    else:
        url = get_asset_url(image) + ":getPixels"

    chip_response = session.post(url, json.dumps(query))

    chip = np.load(io.BytesIO(chip_response.content)).astype("float32")
    # Or pull out nodatas somehow
    return np.where(chip < 0.0, 0.0, chip)


def get_chips(
    pt_tf, feature_image, feature_bands, label_image, label_bands, scale, session
):
    feature_chip = get_chip(
        pt_tf.numpy().tolist(), feature_image, feature_bands, scale, session
    )

    label_chip = get_chip(
        pt_tf.numpy().tolist(), label_image, label_bands, scale, session
    )

    return (
        np.expand_dims(np.expand_dims(feature_chip, axis=0), axis=-1),
        np.expand_dims(np.expand_dims(label_chip, axis=0), axis=-1),
    )


def get_points(n=100):
    countries = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level0")
    germany = countries.filter(ee.Filter.eq("ADM0_NAME", "Germany"))
    pts = ee.FeatureCollection.randomPoints(region=germany, points=n)
    return tf.convert_to_tensor(pts.geometry().coordinates().getInfo())
