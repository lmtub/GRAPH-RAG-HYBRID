void hmp_info_spice(Monitor *mon, const QDict *qdict)

{

    SpiceChannelList *chan;

    SpiceInfo *info;

    const char *channel_name;

    const char * const channel_names[] = {

        [SPICE_CHANNEL_MAIN] = "main",

        [SPICE_CHANNEL_DISPLAY] = "display",

        [SPICE_CHANNEL_INPUTS] = "inputs",

        [SPICE_CHANNEL_CURSOR] = "cursor",

        [SPICE_CHANNEL_PLAYBACK] = "playback",

        [SPICE_CHANNEL_RECORD] = "record",

        [SPICE_CHANNEL_TUNNEL] = "tunnel",

        [SPICE_CHANNEL_SMARTCARD] = "smartcard",

        [SPICE_CHANNEL_USBREDIR] = "usbredir",

        [SPICE_CHANNEL_PORT] = "port",

#if 0

        /* minimum spice-protocol is 0.12.3, webdav was added in 0.12.7,

         * no easy way to #ifdef (SPICE_CHANNEL_* is a enum).  Disable

         * as quick fix for build failures with older versions. */

        [SPICE_CHANNEL_WEBDAV] = "webdav",

#endif

    };



    info = qmp_query_spice(NULL);



    if (!info->enabled) {

        monitor_printf(mon, "Server: disabled\n");

        goto out;

    }



    monitor_printf(mon, "Server:\n");

    if (info->has_port) {

        monitor_printf(mon, "     address: %s:%" PRId64 "\n",

                       info->host, info->port);

    }

    if (info->has_tls_port) {

        monitor_printf(mon, "     address: %s:%" PRId64 " [tls]\n",

                       info->host, info->tls_port);

    }

    monitor_printf(mon, "    migrated: %s\n",

                   info->migrated ? "true" : "false");

    monitor_printf(mon, "        auth: %s\n", info->auth);

    monitor_printf(mon, "    compiled: %s\n", info->compiled_version);

    monitor_printf(mon, "  mouse-mode: %s\n",

                   SpiceQueryMouseMode_lookup[info->mouse_mode]);



    if (!info->has_channels || info->channels == NULL) {

        monitor_printf(mon, "Channels: none\n");

    } else {

        for (chan = info->channels; chan; chan = chan->next) {

            monitor_printf(mon, "Channel:\n");

            monitor_printf(mon, "     address: %s:%s%s\n",

                           chan->value->base->host, chan->value->base->port,

                           chan->value->tls ? " [tls]" : "");

            monitor_printf(mon, "     session: %" PRId64 "\n",

                           chan->value->connection_id);

            monitor_printf(mon, "     channel: %" PRId64 ":%" PRId64 "\n",

                           chan->value->channel_type, chan->value->channel_id);



            channel_name = "unknown";

            if (chan->value->channel_type > 0 &&

                chan->value->channel_type < ARRAY_SIZE(channel_names) &&

                channel_names[chan->value->channel_type]) {

                channel_name = channel_names[chan->value->channel_type];

            }



            monitor_printf(mon, "     channel name: %s\n", channel_name);

        }

    }



out:

    qapi_free_SpiceInfo(info);

}
