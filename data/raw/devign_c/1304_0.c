static void slirp_init_once(void)

{

    static int initialized;

    struct hostent *he;

    char our_name[256];

#ifdef _WIN32

    WSADATA Data;

#endif



    if (initialized) {

        return;

    }

    initialized = 1;



#ifdef _WIN32

    WSAStartup(MAKEWORD(2,0), &Data);

    atexit(winsock_cleanup);

#endif



    loopback_addr.s_addr = htonl(INADDR_LOOPBACK);



    /* FIXME: This address may change during runtime */

    if (gethostname(our_name, sizeof(our_name)) == 0) {

        he = gethostbyname(our_name);

        if (he) {

            our_addr = *(struct in_addr *)he->h_addr;

        }

    }

    if (our_addr.s_addr == 0) {

        our_addr = loopback_addr;

    }



    /* FIXME: This address may change during runtime */

    if (get_dns_addr(&dns_addr) < 0) {

        dns_addr = loopback_addr;

    }

}
