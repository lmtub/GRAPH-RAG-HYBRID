static CharDriverState *vc_init(const char *id, ChardevBackend *backend,

                                ChardevReturn *ret, Error **errp)

{

    return vc_handler(backend->u.vc, errp);

}
