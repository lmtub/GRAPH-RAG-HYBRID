static SpiceChannelList *qmp_query_spice_channels(void)

{

    SpiceChannelList *cur_item = NULL, *head = NULL;

    ChannelList *item;



    QTAILQ_FOREACH(item, &channel_list, link) {

        SpiceChannelList *chan;

        char host[NI_MAXHOST], port[NI_MAXSERV];

        struct sockaddr *paddr;

        socklen_t plen;



        if (!(item->info->flags & SPICE_CHANNEL_EVENT_FLAG_ADDR_EXT)) {

            error_report("invalid channel event");

            return NULL;

        }



        chan = g_malloc0(sizeof(*chan));

        chan->value = g_malloc0(sizeof(*chan->value));

        chan->value->base = g_malloc0(sizeof(*chan->value->base));



        paddr = (struct sockaddr *)&item->info->paddr_ext;

        plen = item->info->plen_ext;

        getnameinfo(paddr, plen,

                    host, sizeof(host), port, sizeof(port),

                    NI_NUMERICHOST | NI_NUMERICSERV);

        chan->value->base->host = g_strdup(host);

        chan->value->base->port = g_strdup(port);

        chan->value->base->family = inet_netfamily(paddr->sa_family);



        chan->value->connection_id = item->info->connection_id;

        chan->value->channel_type = item->info->type;

        chan->value->channel_id = item->info->id;

        chan->value->tls = item->info->flags & SPICE_CHANNEL_EVENT_FLAG_TLS;



       /* XXX: waiting for the qapi to support GSList */

        if (!cur_item) {

            head = cur_item = chan;

        } else {

            cur_item->next = chan;

            cur_item = chan;

        }

    }



    return head;

}
