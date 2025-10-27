static int ipoctal_init(IPackDevice *ip)

{

    IPOctalState *s = IPOCTAL(ip);

    unsigned i;



    for (i = 0; i < N_CHANNELS; i++) {

        SCC2698Channel *ch = &s->ch[i];

        ch->ipoctal = s;



        /* Redirect IP-Octal channels to host character devices */

        if (ch->devpath) {

            const char chr_name[] = "ipoctal";

            char label[ARRAY_SIZE(chr_name) + 2];

            static int index;



            snprintf(label, sizeof(label), "%s%d", chr_name, index);



            ch->dev = qemu_chr_new(label, ch->devpath, NULL);



            if (ch->dev) {

                index++;


                qemu_chr_add_handlers(ch->dev, hostdev_can_receive,

                                      hostdev_receive, hostdev_event, ch);

                DPRINTF("Redirecting channel %u to %s (%s)\n",

                        i, ch->devpath, label);

            } else {

                DPRINTF("Could not redirect channel %u to %s\n",

                        i, ch->devpath);

            }

        }

    }



    return 0;

}