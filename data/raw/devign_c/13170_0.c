static void write_palette(const char *key, QObject *obj, void *opaque)

{

    struct palette_cb_priv *priv = opaque;

    VncState *vs = priv->vs;

    uint32_t bytes = vs->clientds.pf.bytes_per_pixel;

    uint8_t idx = qint_get_int(qobject_to_qint(obj));



    if (bytes == 4) {

        uint32_t color = tight_palette_buf2rgb(32, (uint8_t *)key);



        ((uint32_t*)priv->header)[idx] = color;

    } else {

        uint16_t color = tight_palette_buf2rgb(16, (uint8_t *)key);



        ((uint16_t*)priv->header)[idx] = color;

    }

}
