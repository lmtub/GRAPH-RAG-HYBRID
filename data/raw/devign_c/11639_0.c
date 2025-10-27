void slirp_select_poll(fd_set *readfds, fd_set *writefds, fd_set *xfds)

{

    struct socket *so, *so_next;

    int ret;



    global_readfds = readfds;

    global_writefds = writefds;

    global_xfds = xfds;



	/* Update time */

	updtime();



	/*

	 * See if anything has timed out

	 */

	if (link_up) {

		if (time_fasttimo && ((curtime - time_fasttimo) >= 2)) {

			tcp_fasttimo();

			time_fasttimo = 0;

		}

		if (do_slowtimo && ((curtime - last_slowtimo) >= 499)) {

			ip_slowtimo();

			tcp_slowtimo();

			last_slowtimo = curtime;

		}

	}



	/*

	 * Check sockets

	 */

	if (link_up) {

		/*

		 * Check TCP sockets

		 */

		for (so = tcb.so_next; so != &tcb; so = so_next) {

			so_next = so->so_next;



			/*

			 * FD_ISSET is meaningless on these sockets

			 * (and they can crash the program)

			 */

			if (so->so_state & SS_NOFDREF || so->s == -1)

			   continue;



			/*

			 * Check for URG data

			 * This will soread as well, so no need to

			 * test for readfds below if this succeeds

			 */

			if (FD_ISSET(so->s, xfds))

			   sorecvoob(so);

			/*

			 * Check sockets for reading

			 */

			else if (FD_ISSET(so->s, readfds)) {

				/*

				 * Check for incoming connections

				 */

				if (so->so_state & SS_FACCEPTCONN) {

					tcp_connect(so);

					continue;

				} /* else */

				ret = soread(so);



				/* Output it if we read something */

				if (ret > 0)

				   tcp_output(sototcpcb(so));

			}



			/*

			 * Check sockets for writing

			 */

			if (FD_ISSET(so->s, writefds)) {

			  /*

			   * Check for non-blocking, still-connecting sockets

			   */

			  if (so->so_state & SS_ISFCONNECTING) {

			    /* Connected */

			    so->so_state &= ~SS_ISFCONNECTING;



			    ret = send(so->s, (const void *) &ret, 0, 0);

			    if (ret < 0) {

			      /* XXXXX Must fix, zero bytes is a NOP */

			      if (errno == EAGAIN || errno == EWOULDBLOCK ||

				  errno == EINPROGRESS || errno == ENOTCONN)

				continue;



			      /* else failed */

			      so->so_state &= SS_PERSISTENT_MASK;

			      so->so_state |= SS_NOFDREF;

			    }

			    /* else so->so_state &= ~SS_ISFCONNECTING; */



			    /*

			     * Continue tcp_input

			     */

			    tcp_input((struct mbuf *)NULL, sizeof(struct ip), so);

			    /* continue; */

			  } else

			    ret = sowrite(so);

			  /*

			   * XXXXX If we wrote something (a lot), there

			   * could be a need for a window update.

			   * In the worst case, the remote will send

			   * a window probe to get things going again

			   */

			}



			/*

			 * Probe a still-connecting, non-blocking socket

			 * to check if it's still alive

	 	 	 */

#ifdef PROBE_CONN

			if (so->so_state & SS_ISFCONNECTING) {

			  ret = recv(so->s, (char *)&ret, 0,0);



			  if (ret < 0) {

			    /* XXX */

			    if (errno == EAGAIN || errno == EWOULDBLOCK ||

				errno == EINPROGRESS || errno == ENOTCONN)

			      continue; /* Still connecting, continue */



			    /* else failed */

			    so->so_state &= SS_PERSISTENT_MASK;

			    so->so_state |= SS_NOFDREF;



			    /* tcp_input will take care of it */

			  } else {

			    ret = send(so->s, &ret, 0,0);

			    if (ret < 0) {

			      /* XXX */

			      if (errno == EAGAIN || errno == EWOULDBLOCK ||

				  errno == EINPROGRESS || errno == ENOTCONN)

				continue;

			      /* else failed */

			      so->so_state &= SS_PERSISTENT_MASK;

			      so->so_state |= SS_NOFDREF;

			    } else

			      so->so_state &= ~SS_ISFCONNECTING;



			  }

			  tcp_input((struct mbuf *)NULL, sizeof(struct ip),so);

			} /* SS_ISFCONNECTING */

#endif

		}



		/*

		 * Now UDP sockets.

		 * Incoming packets are sent straight away, they're not buffered.

		 * Incoming UDP data isn't buffered either.

		 */

		for (so = udb.so_next; so != &udb; so = so_next) {

			so_next = so->so_next;



			if (so->s != -1 && FD_ISSET(so->s, readfds)) {

                            sorecvfrom(so);

                        }

		}

	}



	/*

	 * See if we can start outputting

	 */

	if (if_queued && link_up)

	   if_start();



	/* clear global file descriptor sets.

	 * these reside on the stack in vl.c

	 * so they're unusable if we're not in

	 * slirp_select_fill or slirp_select_poll.

	 */

	 global_readfds = NULL;

	 global_writefds = NULL;

	 global_xfds = NULL;

}
