void rng_backend_open(RngBackend *s, Error **errp)

{

    object_property_set_bool(OBJECT(s), true, "opened", errp);

}
