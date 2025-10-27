fork_exec(struct socket *so, const char *ex, int do_pty)

{

	int s;

	struct sockaddr_in addr;

	int addrlen = sizeof(addr);

	int opt;

        int master = -1;

	char *argv[256];

#if 0

	char buff[256];

#endif

	/* don't want to clobber the original */

	char *bptr;

	const char *curarg;

	int c, i, ret;



	DEBUG_CALL("fork_exec");

	DEBUG_ARG("so = %lx", (long)so);

	DEBUG_ARG("ex = %lx", (long)ex);

	DEBUG_ARG("do_pty = %lx", (long)do_pty);



	if (do_pty == 2) {

#if 0

		if (slirp_openpty(&master, &s) == -1) {

			lprint("Error: openpty failed: %s\n", strerror(errno));

			return 0;

		}

#else

                return 0;

#endif

	} else {

		addr.sin_family = AF_INET;

		addr.sin_port = 0;

		addr.sin_addr.s_addr = INADDR_ANY;



		if ((s = socket(AF_INET, SOCK_STREAM, 0)) < 0 ||

		    bind(s, (struct sockaddr *)&addr, addrlen) < 0 ||

		    listen(s, 1) < 0) {

			lprint("Error: inet socket: %s\n", strerror(errno));

			closesocket(s);



			return 0;

		}

	}



	switch(fork()) {

	 case -1:

		lprint("Error: fork failed: %s\n", strerror(errno));

		close(s);

		if (do_pty == 2)

		   close(master);

		return 0;



	 case 0:

		/* Set the DISPLAY */

		if (do_pty == 2) {

			(void) close(master);

#ifdef TIOCSCTTY /* XXXXX */

			(void) setsid();

			ioctl(s, TIOCSCTTY, (char *)NULL);

#endif

		} else {

			getsockname(s, (struct sockaddr *)&addr, &addrlen);

			close(s);

			/*

			 * Connect to the socket

			 * XXX If any of these fail, we're in trouble!

	 		 */

			s = socket(AF_INET, SOCK_STREAM, 0);

			addr.sin_addr = loopback_addr;

                        do {

                            ret = connect(s, (struct sockaddr *)&addr, addrlen);

                        } while (ret < 0 && errno == EINTR);

		}



#if 0

		if (x_port >= 0) {

#ifdef HAVE_SETENV

			sprintf(buff, "%s:%d.%d", inet_ntoa(our_addr), x_port, x_screen);

			setenv("DISPLAY", buff, 1);

#else

			sprintf(buff, "DISPLAY=%s:%d.%d", inet_ntoa(our_addr), x_port, x_screen);

			putenv(buff);

#endif

		}

#endif

		dup2(s, 0);

		dup2(s, 1);

		dup2(s, 2);

		for (s = getdtablesize() - 1; s >= 3; s--)

		   close(s);



		i = 0;

		bptr = strdup(ex); /* No need to free() this */

		if (do_pty == 1) {

			/* Setup "slirp.telnetd -x" */

			argv[i++] = "slirp.telnetd";

			argv[i++] = "-x";

			argv[i++] = bptr;

		} else

		   do {

			/* Change the string into argv[] */

			curarg = bptr;

			while (*bptr != ' ' && *bptr != (char)0)

			   bptr++;

			c = *bptr;

			*bptr++ = (char)0;

			argv[i++] = strdup(curarg);

		   } while (c);



		argv[i] = 0;

		execvp(argv[0], argv);



		/* Ooops, failed, let's tell the user why */

		  {

			  char buff[256];



			  sprintf(buff, "Error: execvp of %s failed: %s\n",

				  argv[0], strerror(errno));

			  write(2, buff, strlen(buff)+1);

		  }

		close(0); close(1); close(2); /* XXX */

		exit(1);



	 default:

		if (do_pty == 2) {

			close(s);

			so->s = master;

		} else {

			/*

			 * XXX this could block us...

			 * XXX Should set a timer here, and if accept() doesn't

		 	 * return after X seconds, declare it a failure

		 	 * The only reason this will block forever is if socket()

		 	 * of connect() fail in the child process

		 	 */

                        do {

                            so->s = accept(s, (struct sockaddr *)&addr, &addrlen);

                        } while (so->s < 0 && errno == EINTR);

                        closesocket(s);

			opt = 1;

			setsockopt(so->s,SOL_SOCKET,SO_REUSEADDR,(char *)&opt,sizeof(int));

			opt = 1;

			setsockopt(so->s,SOL_SOCKET,SO_OOBINLINE,(char *)&opt,sizeof(int));

		}

		fd_nonblock(so->s);



		/* Append the telnet options now */

		if (so->so_m != 0 && do_pty == 1)  {

			sbappend(so, so->so_m);

			so->so_m = 0;

		}



		return 1;

	}

}
