void set_link_completion(ReadLineState *rs, int nb_args, const char *str)

{

    size_t len;



    len = strlen(str);

    readline_set_completion_index(rs, len);

    if (nb_args == 2) {

        NetClientState *ncs[MAX_QUEUE_NUM];

        int count, i;

        count = qemu_find_net_clients_except(NULL, ncs,

                                             NET_CLIENT_OPTIONS_KIND_NONE,

                                             MAX_QUEUE_NUM);

        for (i = 0; i < count; i++) {

            const char *name = ncs[i]->name;

            if (!strncmp(str, name, len)) {

                readline_add_completion(rs, name);

            }

        }

    } else if (nb_args == 3) {

        add_completion_option(rs, str, "on");

        add_completion_option(rs, str, "off");

    }

}
