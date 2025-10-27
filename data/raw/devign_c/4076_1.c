static int resize_peers(IVShmemState *s, int new_min_size)

{



    int j, old_size;



    /* limit number of max peers */

    if (new_min_size <= 0 || new_min_size > IVSHMEM_MAX_PEERS) {

        return -1;

    }

    if (new_min_size <= s->nb_peers) {

        return 0;

    }



    old_size = s->nb_peers;

    s->nb_peers = new_min_size;



    IVSHMEM_DPRINTF("bumping storage to %d peers\n", s->nb_peers);



    s->peers = g_realloc(s->peers, s->nb_peers * sizeof(Peer));



    for (j = old_size; j < s->nb_peers; j++) {

        s->peers[j].eventfds = g_new0(EventNotifier, s->vectors);

        s->peers[j].nb_eventfds = 0;

    }



    return 0;

}
