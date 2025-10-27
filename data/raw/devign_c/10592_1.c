getouraddr(void)

{

	char buff[256];

	struct hostent *he = NULL;



	if (gethostname(buff,256) == 0)

            he = gethostbyname(buff);

        if (he)

            our_addr = *(struct in_addr *)he->h_addr;

        if (our_addr.s_addr == 0)

            our_addr.s_addr = loopback_addr.s_addr;

}
