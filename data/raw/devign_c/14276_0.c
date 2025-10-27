virtio_crypto_check_cryptodev_is_used(const Object *obj, const char *name,

                                      Object *val, Error **errp)

{

    if (cryptodev_backend_is_used(CRYPTODEV_BACKEND(val))) {

        char *path = object_get_canonical_path_component(val);

        error_setg(errp,

            "can't use already used cryptodev backend: %s", path);

        g_free(path);

    } else {

        qdev_prop_allow_set_link_before_realize(obj, name, val, errp);

    }

}
