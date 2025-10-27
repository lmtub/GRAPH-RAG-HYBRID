tcp_dooptions(struct tcpcb *tp, u_char *cp, int cnt, struct tcpiphdr *ti)

{

	uint16_t mss;

	int opt, optlen;



	DEBUG_CALL("tcp_dooptions");

	DEBUG_ARGS((dfd," tp = %lx  cnt=%i \n", (long )tp, cnt));



	for (; cnt > 0; cnt -= optlen, cp += optlen) {

		opt = cp[0];

		if (opt == TCPOPT_EOL)

			break;

		if (opt == TCPOPT_NOP)

			optlen = 1;

		else {

			optlen = cp[1];

			if (optlen <= 0)

				break;

		}

		switch (opt) {



		default:

			continue;



		case TCPOPT_MAXSEG:

			if (optlen != TCPOLEN_MAXSEG)

				continue;

			if (!(ti->ti_flags & TH_SYN))

				continue;

			memcpy((char *) &mss, (char *) cp + 2, sizeof(mss));

			NTOHS(mss);

			(void) tcp_mss(tp, mss);	/* sets t_maxseg */

			break;

		}

	}

}
