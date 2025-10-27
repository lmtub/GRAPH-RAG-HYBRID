static int vhost_user_start(VhostUserState *s)

{

    VhostNetOptions options;



    if (vhost_user_running(s)) {

        return 0;

    }



    options.backend_type = VHOST_BACKEND_TYPE_USER;

    options.net_backend = &s->nc;

    options.opaque = s->chr;

    options.force = true;



    s->vhost_net = vhost_net_init(&options);



    return vhost_user_running(s) ? 0 : -1;

}
