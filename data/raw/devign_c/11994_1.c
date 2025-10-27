qcrypto_tls_creds_x509_init(Object *obj)
{
    object_property_add_bool(obj, "loaded",
                             qcrypto_tls_creds_x509_prop_get_loaded,
                             qcrypto_tls_creds_x509_prop_set_loaded,
}