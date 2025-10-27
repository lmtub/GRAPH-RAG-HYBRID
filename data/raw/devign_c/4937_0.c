static int get_dns_addr(struct in_addr *pdns_addr)

{

    char buff[512];

    char buff2[257];

    FILE *f;

    int found = 0;

    struct in_addr tmp_addr;



    f = fopen("/etc/resolv.conf", "r");

    if (!f)

        return -1;



#ifdef DEBUG

    lprint("IP address of your DNS(s): ");

#endif

    while (fgets(buff, 512, f) != NULL) {

        if (sscanf(buff, "nameserver%*[ \t]%256s", buff2) == 1) {

            if (!inet_aton(buff2, &tmp_addr))

                continue;

            if (tmp_addr.s_addr == loopback_addr.s_addr)

                tmp_addr = our_addr;

            /* If it's the first one, set it to dns_addr */

            if (!found)

                *pdns_addr = tmp_addr;

#ifdef DEBUG

            else

                lprint(", ");

#endif

            if (++found > 3) {

#ifdef DEBUG

                lprint("(more)");

#endif

                break;

            }

#ifdef DEBUG

            else

                lprint("%s", inet_ntoa(tmp_addr));

#endif

        }

    }

    fclose(f);

    if (!found)

        return -1;

    return 0;

}
