static void net_vhost_link_down(VhostUserState *s, bool link_down)

{

    s->nc.link_down = link_down;



    if (s->nc.peer) {

        s->nc.peer->link_down = link_down;

    }



    if (s->nc.info->link_status_changed) {

        s->nc.info->link_status_changed(&s->nc);

    }



    if (s->nc.peer && s->nc.peer->info->link_status_changed) {

        s->nc.peer->info->link_status_changed(s->nc.peer);

    }

}
