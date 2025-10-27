tcp_input(struct mbuf *m, int iphlen, struct socket *inso)

{

  	struct ip save_ip, *ip;

	register struct tcpiphdr *ti;

	caddr_t optp = NULL;

	int optlen = 0;

	int len, tlen, off;

        register struct tcpcb *tp = NULL;

	register int tiflags;

        struct socket *so = NULL;

	int todrop, acked, ourfinisacked, needoutput = 0;

	int iss = 0;

	u_long tiwin;

	int ret;

    struct ex_list *ex_ptr;

    Slirp *slirp;



	DEBUG_CALL("tcp_input");

	DEBUG_ARGS((dfd, " m = %8lx  iphlen = %2d  inso = %lx\n",

		    (long )m, iphlen, (long )inso ));



	/*

	 * If called with m == 0, then we're continuing the connect

	 */

	if (m == NULL) {

		so = inso;

		slirp = so->slirp;



		/* Re-set a few variables */

		tp = sototcpcb(so);

		m = so->so_m;

                so->so_m = NULL;

		ti = so->so_ti;

		tiwin = ti->ti_win;

		tiflags = ti->ti_flags;



		goto cont_conn;

	}

	slirp = m->slirp;



	/*

	 * Get IP and TCP header together in first mbuf.

	 * Note: IP leaves IP header in first mbuf.

	 */

	ti = mtod(m, struct tcpiphdr *);

	if (iphlen > sizeof(struct ip )) {

	  ip_stripoptions(m, (struct mbuf *)0);

	  iphlen=sizeof(struct ip );

	}

	/* XXX Check if too short */





	/*

	 * Save a copy of the IP header in case we want restore it

	 * for sending an ICMP error message in response.

	 */

	ip=mtod(m, struct ip *);

	save_ip = *ip;

	save_ip.ip_len+= iphlen;



	/*

	 * Checksum extended TCP header and data.

	 */

	tlen = ((struct ip *)ti)->ip_len;

        tcpiphdr2qlink(ti)->next = tcpiphdr2qlink(ti)->prev = NULL;

        memset(&ti->ti_i.ih_mbuf, 0 , sizeof(struct mbuf_ptr));

	ti->ti_x1 = 0;

	ti->ti_len = htons((uint16_t)tlen);

	len = sizeof(struct ip ) + tlen;

	if(cksum(m, len)) {

	  goto drop;

	}



	/*

	 * Check that TCP offset makes sense,

	 * pull out TCP options and adjust length.		XXX

	 */

	off = ti->ti_off << 2;

	if (off < sizeof (struct tcphdr) || off > tlen) {

	  goto drop;

	}

	tlen -= off;

	ti->ti_len = tlen;

	if (off > sizeof (struct tcphdr)) {

	  optlen = off - sizeof (struct tcphdr);

	  optp = mtod(m, caddr_t) + sizeof (struct tcpiphdr);

	}

	tiflags = ti->ti_flags;



	/*

	 * Convert TCP protocol specific fields to host format.

	 */

	NTOHL(ti->ti_seq);

	NTOHL(ti->ti_ack);

	NTOHS(ti->ti_win);

	NTOHS(ti->ti_urp);



	/*

	 * Drop TCP, IP headers and TCP options.

	 */

	m->m_data += sizeof(struct tcpiphdr)+off-sizeof(struct tcphdr);

	m->m_len  -= sizeof(struct tcpiphdr)+off-sizeof(struct tcphdr);



    if (slirp->restricted) {

        for (ex_ptr = slirp->exec_list; ex_ptr; ex_ptr = ex_ptr->ex_next) {

            if (ex_ptr->ex_fport == ti->ti_dport &&

                ti->ti_dst.s_addr == ex_ptr->ex_addr.s_addr) {

                break;

            }

        }

        if (!ex_ptr)

            goto drop;

    }

	/*

	 * Locate pcb for segment.

	 */

findso:

	so = slirp->tcp_last_so;

	if (so->so_fport != ti->ti_dport ||

	    so->so_lport != ti->ti_sport ||

	    so->so_laddr.s_addr != ti->ti_src.s_addr ||

	    so->so_faddr.s_addr != ti->ti_dst.s_addr) {

		so = solookup(&slirp->tcb, ti->ti_src, ti->ti_sport,

			       ti->ti_dst, ti->ti_dport);

		if (so)

			slirp->tcp_last_so = so;

	}



	/*

	 * If the state is CLOSED (i.e., TCB does not exist) then

	 * all data in the incoming segment is discarded.

	 * If the TCB exists but is in CLOSED state, it is embryonic,

	 * but should either do a listen or a connect soon.

	 *

	 * state == CLOSED means we've done socreate() but haven't

	 * attached it to a protocol yet...

	 *

	 * XXX If a TCB does not exist, and the TH_SYN flag is

	 * the only flag set, then create a session, mark it

	 * as if it was LISTENING, and continue...

	 */

        if (so == NULL) {

	  if ((tiflags & (TH_SYN|TH_FIN|TH_RST|TH_URG|TH_ACK)) != TH_SYN)

	    goto dropwithreset;



	  if ((so = socreate(slirp)) == NULL)

	    goto dropwithreset;

	  if (tcp_attach(so) < 0) {

	    free(so); /* Not sofree (if it failed, it's not insqued) */

	    goto dropwithreset;

	  }



	  sbreserve(&so->so_snd, TCP_SNDSPACE);

	  sbreserve(&so->so_rcv, TCP_RCVSPACE);



	  so->so_laddr = ti->ti_src;

	  so->so_lport = ti->ti_sport;

	  so->so_faddr = ti->ti_dst;

	  so->so_fport = ti->ti_dport;



	  if ((so->so_iptos = tcp_tos(so)) == 0)

	    so->so_iptos = ((struct ip *)ti)->ip_tos;



	  tp = sototcpcb(so);

	  tp->t_state = TCPS_LISTEN;

	}



        /*

         * If this is a still-connecting socket, this probably

         * a retransmit of the SYN.  Whether it's a retransmit SYN

	 * or something else, we nuke it.

         */

        if (so->so_state & SS_ISFCONNECTING)

                goto drop;



	tp = sototcpcb(so);



	/* XXX Should never fail */

        if (tp == NULL)

		goto dropwithreset;

	if (tp->t_state == TCPS_CLOSED)

		goto drop;



	tiwin = ti->ti_win;



	/*

	 * Segment received on connection.

	 * Reset idle time and keep-alive timer.

	 */

	tp->t_idle = 0;

	if (SO_OPTIONS)

	   tp->t_timer[TCPT_KEEP] = TCPTV_KEEPINTVL;

	else

	   tp->t_timer[TCPT_KEEP] = TCPTV_KEEP_IDLE;



	/*

	 * Process options if not in LISTEN state,

	 * else do it below (after getting remote address).

	 */

	if (optp && tp->t_state != TCPS_LISTEN)

		tcp_dooptions(tp, (u_char *)optp, optlen, ti);



	/*

	 * Header prediction: check for the two common cases

	 * of a uni-directional data xfer.  If the packet has

	 * no control flags, is in-sequence, the window didn't

	 * change and we're not retransmitting, it's a

	 * candidate.  If the length is zero and the ack moved

	 * forward, we're the sender side of the xfer.  Just

	 * free the data acked & wake any higher level process

	 * that was blocked waiting for space.  If the length

	 * is non-zero and the ack didn't move, we're the

	 * receiver side.  If we're getting packets in-order

	 * (the reassembly queue is empty), add the data to

	 * the socket buffer and note that we need a delayed ack.

	 *

	 * XXX Some of these tests are not needed

	 * eg: the tiwin == tp->snd_wnd prevents many more

	 * predictions.. with no *real* advantage..

	 */

	if (tp->t_state == TCPS_ESTABLISHED &&

	    (tiflags & (TH_SYN|TH_FIN|TH_RST|TH_URG|TH_ACK)) == TH_ACK &&

	    ti->ti_seq == tp->rcv_nxt &&

	    tiwin && tiwin == tp->snd_wnd &&

	    tp->snd_nxt == tp->snd_max) {

		if (ti->ti_len == 0) {

			if (SEQ_GT(ti->ti_ack, tp->snd_una) &&

			    SEQ_LEQ(ti->ti_ack, tp->snd_max) &&

			    tp->snd_cwnd >= tp->snd_wnd) {

				/*

				 * this is a pure ack for outstanding data.

				 */

				if (tp->t_rtt &&

				    SEQ_GT(ti->ti_ack, tp->t_rtseq))

					tcp_xmit_timer(tp, tp->t_rtt);

				acked = ti->ti_ack - tp->snd_una;

				sbdrop(&so->so_snd, acked);

				tp->snd_una = ti->ti_ack;

				m_free(m);



				/*

				 * If all outstanding data are acked, stop

				 * retransmit timer, otherwise restart timer

				 * using current (possibly backed-off) value.

				 * If process is waiting for space,

				 * wakeup/selwakeup/signal.  If data

				 * are ready to send, let tcp_output

				 * decide between more output or persist.

				 */

				if (tp->snd_una == tp->snd_max)

					tp->t_timer[TCPT_REXMT] = 0;

				else if (tp->t_timer[TCPT_PERSIST] == 0)

					tp->t_timer[TCPT_REXMT] = tp->t_rxtcur;



				/*

				 * This is called because sowwakeup might have

				 * put data into so_snd.  Since we don't so sowwakeup,

				 * we don't need this.. XXX???

				 */

				if (so->so_snd.sb_cc)

					(void) tcp_output(tp);



				return;

			}

		} else if (ti->ti_ack == tp->snd_una &&

		    tcpfrag_list_empty(tp) &&

		    ti->ti_len <= sbspace(&so->so_rcv)) {

			/*

			 * this is a pure, in-sequence data packet

			 * with nothing on the reassembly queue and

			 * we have enough buffer space to take it.

			 */

			tp->rcv_nxt += ti->ti_len;

			/*

			 * Add data to socket buffer.

			 */

			if (so->so_emu) {

				if (tcp_emu(so,m)) sbappend(so, m);

			} else

				sbappend(so, m);



			/*

			 * If this is a short packet, then ACK now - with Nagel

			 *	congestion avoidance sender won't send more until

			 *	he gets an ACK.

			 *

			 * It is better to not delay acks at all to maximize

			 * TCP throughput.  See RFC 2581.

			 */

			tp->t_flags |= TF_ACKNOW;

			tcp_output(tp);

			return;

		}

	} /* header prediction */

	/*

	 * Calculate amount of space in receive window,

	 * and then do TCP input processing.

	 * Receive window is amount of space in rcv queue,

	 * but not less than advertised window.

	 */

	{ int win;

          win = sbspace(&so->so_rcv);

	  if (win < 0)

	    win = 0;

	  tp->rcv_wnd = max(win, (int)(tp->rcv_adv - tp->rcv_nxt));

	}



	switch (tp->t_state) {



	/*

	 * If the state is LISTEN then ignore segment if it contains an RST.

	 * If the segment contains an ACK then it is bad and send a RST.

	 * If it does not contain a SYN then it is not interesting; drop it.

	 * Don't bother responding if the destination was a broadcast.

	 * Otherwise initialize tp->rcv_nxt, and tp->irs, select an initial

	 * tp->iss, and send a segment:

	 *     <SEQ=ISS><ACK=RCV_NXT><CTL=SYN,ACK>

	 * Also initialize tp->snd_nxt to tp->iss+1 and tp->snd_una to tp->iss.

	 * Fill in remote peer address fields if not previously specified.

	 * Enter SYN_RECEIVED state, and process any other fields of this

	 * segment in this state.

	 */

	case TCPS_LISTEN: {



	  if (tiflags & TH_RST)

	    goto drop;

	  if (tiflags & TH_ACK)

	    goto dropwithreset;

	  if ((tiflags & TH_SYN) == 0)

	    goto drop;



	  /*

	   * This has way too many gotos...

	   * But a bit of spaghetti code never hurt anybody :)

	   */



	  /*

	   * If this is destined for the control address, then flag to

	   * tcp_ctl once connected, otherwise connect

	   */

	  if ((so->so_faddr.s_addr & slirp->vnetwork_mask.s_addr) ==

	      slirp->vnetwork_addr.s_addr) {

	    if (so->so_faddr.s_addr != slirp->vhost_addr.s_addr &&

		so->so_faddr.s_addr != slirp->vnameserver_addr.s_addr) {

		/* May be an add exec */

		for (ex_ptr = slirp->exec_list; ex_ptr;

		     ex_ptr = ex_ptr->ex_next) {

		  if(ex_ptr->ex_fport == so->so_fport &&

		     so->so_faddr.s_addr == ex_ptr->ex_addr.s_addr) {

		    so->so_state |= SS_CTL;

		    break;

		  }

		}

		if (so->so_state & SS_CTL) {

		    goto cont_input;

		}

	    }

	    /* CTL_ALIAS: Do nothing, tcp_fconnect will be called on it */

	  }



	  if (so->so_emu & EMU_NOCONNECT) {

	    so->so_emu &= ~EMU_NOCONNECT;

	    goto cont_input;

	  }



	  if((tcp_fconnect(so) == -1) && (errno != EINPROGRESS) && (errno != EWOULDBLOCK)) {

	    u_char code=ICMP_UNREACH_NET;

	    DEBUG_MISC((dfd, " tcp fconnect errno = %d-%s\n",

			errno,strerror(errno)));

	    if(errno == ECONNREFUSED) {

	      /* ACK the SYN, send RST to refuse the connection */

	      tcp_respond(tp, ti, m, ti->ti_seq+1, (tcp_seq)0,

			  TH_RST|TH_ACK);

	    } else {

	      if(errno == EHOSTUNREACH) code=ICMP_UNREACH_HOST;

	      HTONL(ti->ti_seq);             /* restore tcp header */

	      HTONL(ti->ti_ack);

	      HTONS(ti->ti_win);

	      HTONS(ti->ti_urp);

	      m->m_data -= sizeof(struct tcpiphdr)+off-sizeof(struct tcphdr);

	      m->m_len  += sizeof(struct tcpiphdr)+off-sizeof(struct tcphdr);

	      *ip=save_ip;

	      icmp_error(m, ICMP_UNREACH,code, 0,strerror(errno));

	    }

            tcp_close(tp);

	    m_free(m);

	  } else {

	    /*

	     * Haven't connected yet, save the current mbuf

	     * and ti, and return

	     * XXX Some OS's don't tell us whether the connect()

	     * succeeded or not.  So we must time it out.

	     */

	    so->so_m = m;

	    so->so_ti = ti;

	    tp->t_timer[TCPT_KEEP] = TCPTV_KEEP_INIT;

	    tp->t_state = TCPS_SYN_RECEIVED;

	    tcp_template(tp);

	  }

	  return;



	cont_conn:

	  /* m==NULL

	   * Check if the connect succeeded

	   */

	  if (so->so_state & SS_NOFDREF) {

	    tp = tcp_close(tp);

	    goto dropwithreset;

	  }

	cont_input:

	  tcp_template(tp);



	  if (optp)

	    tcp_dooptions(tp, (u_char *)optp, optlen, ti);



	  if (iss)

	    tp->iss = iss;

	  else

	    tp->iss = slirp->tcp_iss;

	  slirp->tcp_iss += TCP_ISSINCR/2;

	  tp->irs = ti->ti_seq;

	  tcp_sendseqinit(tp);

	  tcp_rcvseqinit(tp);

	  tp->t_flags |= TF_ACKNOW;

	  tp->t_state = TCPS_SYN_RECEIVED;

	  tp->t_timer[TCPT_KEEP] = TCPTV_KEEP_INIT;

	  goto trimthenstep6;

	} /* case TCPS_LISTEN */



	/*

	 * If the state is SYN_SENT:

	 *	if seg contains an ACK, but not for our SYN, drop the input.

	 *	if seg contains a RST, then drop the connection.

	 *	if seg does not contain SYN, then drop it.

	 * Otherwise this is an acceptable SYN segment

	 *	initialize tp->rcv_nxt and tp->irs

	 *	if seg contains ack then advance tp->snd_una

	 *	if SYN has been acked change to ESTABLISHED else SYN_RCVD state

	 *	arrange for segment to be acked (eventually)

	 *	continue processing rest of data/controls, beginning with URG

	 */

	case TCPS_SYN_SENT:

		if ((tiflags & TH_ACK) &&

		    (SEQ_LEQ(ti->ti_ack, tp->iss) ||

		     SEQ_GT(ti->ti_ack, tp->snd_max)))

			goto dropwithreset;



		if (tiflags & TH_RST) {

                        if (tiflags & TH_ACK) {

                                tcp_drop(tp, 0); /* XXX Check t_softerror! */

                        }

			goto drop;

		}



		if ((tiflags & TH_SYN) == 0)

			goto drop;

		if (tiflags & TH_ACK) {

			tp->snd_una = ti->ti_ack;

			if (SEQ_LT(tp->snd_nxt, tp->snd_una))

				tp->snd_nxt = tp->snd_una;

		}



		tp->t_timer[TCPT_REXMT] = 0;

		tp->irs = ti->ti_seq;

		tcp_rcvseqinit(tp);

		tp->t_flags |= TF_ACKNOW;

		if (tiflags & TH_ACK && SEQ_GT(tp->snd_una, tp->iss)) {

			soisfconnected(so);

			tp->t_state = TCPS_ESTABLISHED;



			(void) tcp_reass(tp, (struct tcpiphdr *)0,

				(struct mbuf *)0);

			/*

			 * if we didn't have to retransmit the SYN,

			 * use its rtt as our initial srtt & rtt var.

			 */

			if (tp->t_rtt)

				tcp_xmit_timer(tp, tp->t_rtt);

		} else

			tp->t_state = TCPS_SYN_RECEIVED;



trimthenstep6:

		/*

		 * Advance ti->ti_seq to correspond to first data byte.

		 * If data, trim to stay within window,

		 * dropping FIN if necessary.

		 */

		ti->ti_seq++;

		if (ti->ti_len > tp->rcv_wnd) {

			todrop = ti->ti_len - tp->rcv_wnd;

			m_adj(m, -todrop);

			ti->ti_len = tp->rcv_wnd;

			tiflags &= ~TH_FIN;

		}

		tp->snd_wl1 = ti->ti_seq - 1;

		tp->rcv_up = ti->ti_seq;

		goto step6;

	} /* switch tp->t_state */

	/*

	 * States other than LISTEN or SYN_SENT.

	 * Check that at least some bytes of segment are within

	 * receive window.  If segment begins before rcv_nxt,

	 * drop leading data (and SYN); if nothing left, just ack.

	 */

	todrop = tp->rcv_nxt - ti->ti_seq;

	if (todrop > 0) {

		if (tiflags & TH_SYN) {

			tiflags &= ~TH_SYN;

			ti->ti_seq++;

			if (ti->ti_urp > 1)

				ti->ti_urp--;

			else

				tiflags &= ~TH_URG;

			todrop--;

		}

		/*

		 * Following if statement from Stevens, vol. 2, p. 960.

		 */

		if (todrop > ti->ti_len

		    || (todrop == ti->ti_len && (tiflags & TH_FIN) == 0)) {

			/*

			 * Any valid FIN must be to the left of the window.

			 * At this point the FIN must be a duplicate or out

			 * of sequence; drop it.

			 */

			tiflags &= ~TH_FIN;



			/*

			 * Send an ACK to resynchronize and drop any data.

			 * But keep on processing for RST or ACK.

			 */

			tp->t_flags |= TF_ACKNOW;

			todrop = ti->ti_len;

		}

		m_adj(m, todrop);

		ti->ti_seq += todrop;

		ti->ti_len -= todrop;

		if (ti->ti_urp > todrop)

			ti->ti_urp -= todrop;

		else {

			tiflags &= ~TH_URG;

			ti->ti_urp = 0;

		}

	}

	/*

	 * If new data are received on a connection after the

	 * user processes are gone, then RST the other end.

	 */

	if ((so->so_state & SS_NOFDREF) &&

	    tp->t_state > TCPS_CLOSE_WAIT && ti->ti_len) {

		tp = tcp_close(tp);

		goto dropwithreset;

	}



	/*

	 * If segment ends after window, drop trailing data

	 * (and PUSH and FIN); if nothing left, just ACK.

	 */

	todrop = (ti->ti_seq+ti->ti_len) - (tp->rcv_nxt+tp->rcv_wnd);

	if (todrop > 0) {

		if (todrop >= ti->ti_len) {

			/*

			 * If a new connection request is received

			 * while in TIME_WAIT, drop the old connection

			 * and start over if the sequence numbers

			 * are above the previous ones.

			 */

			if (tiflags & TH_SYN &&

			    tp->t_state == TCPS_TIME_WAIT &&

			    SEQ_GT(ti->ti_seq, tp->rcv_nxt)) {

				iss = tp->rcv_nxt + TCP_ISSINCR;

				tp = tcp_close(tp);

				goto findso;

			}

			/*

			 * If window is closed can only take segments at

			 * window edge, and have to drop data and PUSH from

			 * incoming segments.  Continue processing, but

			 * remember to ack.  Otherwise, drop segment

			 * and ack.

			 */

			if (tp->rcv_wnd == 0 && ti->ti_seq == tp->rcv_nxt) {

				tp->t_flags |= TF_ACKNOW;

			} else {

				goto dropafterack;

			}

		}

		m_adj(m, -todrop);

		ti->ti_len -= todrop;

		tiflags &= ~(TH_PUSH|TH_FIN);

	}



	/*

	 * If the RST bit is set examine the state:

	 *    SYN_RECEIVED STATE:

	 *	If passive open, return to LISTEN state.

	 *	If active open, inform user that connection was refused.

	 *    ESTABLISHED, FIN_WAIT_1, FIN_WAIT2, CLOSE_WAIT STATES:

	 *	Inform user that connection was reset, and close tcb.

	 *    CLOSING, LAST_ACK, TIME_WAIT STATES

	 *	Close the tcb.

	 */

	if (tiflags&TH_RST) switch (tp->t_state) {



	case TCPS_SYN_RECEIVED:

	case TCPS_ESTABLISHED:

	case TCPS_FIN_WAIT_1:

	case TCPS_FIN_WAIT_2:

	case TCPS_CLOSE_WAIT:

		tp->t_state = TCPS_CLOSED;

                tcp_close(tp);

		goto drop;



	case TCPS_CLOSING:

	case TCPS_LAST_ACK:

	case TCPS_TIME_WAIT:

                tcp_close(tp);

		goto drop;

	}



	/*

	 * If a SYN is in the window, then this is an

	 * error and we send an RST and drop the connection.

	 */

	if (tiflags & TH_SYN) {

		tp = tcp_drop(tp,0);

		goto dropwithreset;

	}



	/*

	 * If the ACK bit is off we drop the segment and return.

	 */

	if ((tiflags & TH_ACK) == 0) goto drop;



	/*

	 * Ack processing.

	 */

	switch (tp->t_state) {

	/*

	 * In SYN_RECEIVED state if the ack ACKs our SYN then enter

	 * ESTABLISHED state and continue processing, otherwise

	 * send an RST.  una<=ack<=max

	 */

	case TCPS_SYN_RECEIVED:



		if (SEQ_GT(tp->snd_una, ti->ti_ack) ||

		    SEQ_GT(ti->ti_ack, tp->snd_max))

			goto dropwithreset;

		tp->t_state = TCPS_ESTABLISHED;

		/*

		 * The sent SYN is ack'ed with our sequence number +1

		 * The first data byte already in the buffer will get

		 * lost if no correction is made.  This is only needed for

		 * SS_CTL since the buffer is empty otherwise.

		 * tp->snd_una++; or:

		 */

		tp->snd_una=ti->ti_ack;

		if (so->so_state & SS_CTL) {

		  /* So tcp_ctl reports the right state */

		  ret = tcp_ctl(so);

		  if (ret == 1) {

		    soisfconnected(so);

		    so->so_state &= ~SS_CTL;   /* success XXX */

		  } else if (ret == 2) {

		    so->so_state &= SS_PERSISTENT_MASK;

		    so->so_state |= SS_NOFDREF; /* CTL_CMD */

		  } else {

		    needoutput = 1;

		    tp->t_state = TCPS_FIN_WAIT_1;

		  }

		} else {

		  soisfconnected(so);

		}



		(void) tcp_reass(tp, (struct tcpiphdr *)0, (struct mbuf *)0);

		tp->snd_wl1 = ti->ti_seq - 1;

		/* Avoid ack processing; snd_una==ti_ack  =>  dup ack */

		goto synrx_to_est;

		/* fall into ... */



	/*

	 * In ESTABLISHED state: drop duplicate ACKs; ACK out of range

	 * ACKs.  If the ack is in the range

	 *	tp->snd_una < ti->ti_ack <= tp->snd_max

	 * then advance tp->snd_una to ti->ti_ack and drop

	 * data from the retransmission queue.  If this ACK reflects

	 * more up to date window information we update our window information.

	 */

	case TCPS_ESTABLISHED:

	case TCPS_FIN_WAIT_1:

	case TCPS_FIN_WAIT_2:

	case TCPS_CLOSE_WAIT:

	case TCPS_CLOSING:

	case TCPS_LAST_ACK:

	case TCPS_TIME_WAIT:



		if (SEQ_LEQ(ti->ti_ack, tp->snd_una)) {

			if (ti->ti_len == 0 && tiwin == tp->snd_wnd) {

			  DEBUG_MISC((dfd, " dup ack  m = %lx  so = %lx\n",

				      (long )m, (long )so));

				/*

				 * If we have outstanding data (other than

				 * a window probe), this is a completely

				 * duplicate ack (ie, window info didn't

				 * change), the ack is the biggest we've

				 * seen and we've seen exactly our rexmt

				 * threshold of them, assume a packet

				 * has been dropped and retransmit it.

				 * Kludge snd_nxt & the congestion

				 * window so we send only this one

				 * packet.

				 *

				 * We know we're losing at the current

				 * window size so do congestion avoidance

				 * (set ssthresh to half the current window

				 * and pull our congestion window back to

				 * the new ssthresh).

				 *

				 * Dup acks mean that packets have left the

				 * network (they're now cached at the receiver)

				 * so bump cwnd by the amount in the receiver

				 * to keep a constant cwnd packets in the

				 * network.

				 */

				if (tp->t_timer[TCPT_REXMT] == 0 ||

				    ti->ti_ack != tp->snd_una)

					tp->t_dupacks = 0;

				else if (++tp->t_dupacks == TCPREXMTTHRESH) {

					tcp_seq onxt = tp->snd_nxt;

					u_int win =

					    min(tp->snd_wnd, tp->snd_cwnd) / 2 /

						tp->t_maxseg;



					if (win < 2)

						win = 2;

					tp->snd_ssthresh = win * tp->t_maxseg;

					tp->t_timer[TCPT_REXMT] = 0;

					tp->t_rtt = 0;

					tp->snd_nxt = ti->ti_ack;

					tp->snd_cwnd = tp->t_maxseg;

					(void) tcp_output(tp);

					tp->snd_cwnd = tp->snd_ssthresh +

					       tp->t_maxseg * tp->t_dupacks;

					if (SEQ_GT(onxt, tp->snd_nxt))

						tp->snd_nxt = onxt;

					goto drop;

				} else if (tp->t_dupacks > TCPREXMTTHRESH) {

					tp->snd_cwnd += tp->t_maxseg;

					(void) tcp_output(tp);

					goto drop;

				}

			} else

				tp->t_dupacks = 0;

			break;

		}

	synrx_to_est:

		/*

		 * If the congestion window was inflated to account

		 * for the other side's cached packets, retract it.

		 */

		if (tp->t_dupacks > TCPREXMTTHRESH &&

		    tp->snd_cwnd > tp->snd_ssthresh)

			tp->snd_cwnd = tp->snd_ssthresh;

		tp->t_dupacks = 0;

		if (SEQ_GT(ti->ti_ack, tp->snd_max)) {

			goto dropafterack;

		}

		acked = ti->ti_ack - tp->snd_una;



		/*

		 * If transmit timer is running and timed sequence

		 * number was acked, update smoothed round trip time.

		 * Since we now have an rtt measurement, cancel the

		 * timer backoff (cf., Phil Karn's retransmit alg.).

		 * Recompute the initial retransmit timer.

		 */

		if (tp->t_rtt && SEQ_GT(ti->ti_ack, tp->t_rtseq))

			tcp_xmit_timer(tp,tp->t_rtt);



		/*

		 * If all outstanding data is acked, stop retransmit

		 * timer and remember to restart (more output or persist).

		 * If there is more data to be acked, restart retransmit

		 * timer, using current (possibly backed-off) value.

		 */

		if (ti->ti_ack == tp->snd_max) {

			tp->t_timer[TCPT_REXMT] = 0;

			needoutput = 1;

		} else if (tp->t_timer[TCPT_PERSIST] == 0)

			tp->t_timer[TCPT_REXMT] = tp->t_rxtcur;

		/*

		 * When new data is acked, open the congestion window.

		 * If the window gives us less than ssthresh packets

		 * in flight, open exponentially (maxseg per packet).

		 * Otherwise open linearly: maxseg per window

		 * (maxseg^2 / cwnd per packet).

		 */

		{

		  register u_int cw = tp->snd_cwnd;

		  register u_int incr = tp->t_maxseg;



		  if (cw > tp->snd_ssthresh)

		    incr = incr * incr / cw;

		  tp->snd_cwnd = min(cw + incr, TCP_MAXWIN<<tp->snd_scale);

		}

		if (acked > so->so_snd.sb_cc) {

			tp->snd_wnd -= so->so_snd.sb_cc;

			sbdrop(&so->so_snd, (int )so->so_snd.sb_cc);

			ourfinisacked = 1;

		} else {

			sbdrop(&so->so_snd, acked);

			tp->snd_wnd -= acked;

			ourfinisacked = 0;

		}

		tp->snd_una = ti->ti_ack;

		if (SEQ_LT(tp->snd_nxt, tp->snd_una))

			tp->snd_nxt = tp->snd_una;



		switch (tp->t_state) {



		/*

		 * In FIN_WAIT_1 STATE in addition to the processing

		 * for the ESTABLISHED state if our FIN is now acknowledged

		 * then enter FIN_WAIT_2.

		 */

		case TCPS_FIN_WAIT_1:

			if (ourfinisacked) {

				/*

				 * If we can't receive any more

				 * data, then closing user can proceed.

				 * Starting the timer is contrary to the

				 * specification, but if we don't get a FIN

				 * we'll hang forever.

				 */

				if (so->so_state & SS_FCANTRCVMORE) {

					tp->t_timer[TCPT_2MSL] = TCP_MAXIDLE;

				}

				tp->t_state = TCPS_FIN_WAIT_2;

			}

			break;



	 	/*

		 * In CLOSING STATE in addition to the processing for

		 * the ESTABLISHED state if the ACK acknowledges our FIN

		 * then enter the TIME-WAIT state, otherwise ignore

		 * the segment.

		 */

		case TCPS_CLOSING:

			if (ourfinisacked) {

				tp->t_state = TCPS_TIME_WAIT;

				tcp_canceltimers(tp);

				tp->t_timer[TCPT_2MSL] = 2 * TCPTV_MSL;

			}

			break;



		/*

		 * In LAST_ACK, we may still be waiting for data to drain

		 * and/or to be acked, as well as for the ack of our FIN.

		 * If our FIN is now acknowledged, delete the TCB,

		 * enter the closed state and return.

		 */

		case TCPS_LAST_ACK:

			if (ourfinisacked) {

                                tcp_close(tp);

				goto drop;

			}

			break;



		/*

		 * In TIME_WAIT state the only thing that should arrive

		 * is a retransmission of the remote FIN.  Acknowledge

		 * it and restart the finack timer.

		 */

		case TCPS_TIME_WAIT:

			tp->t_timer[TCPT_2MSL] = 2 * TCPTV_MSL;

			goto dropafterack;

		}

	} /* switch(tp->t_state) */



step6:

	/*

	 * Update window information.

	 * Don't look at window if no ACK: TAC's send garbage on first SYN.

	 */

	if ((tiflags & TH_ACK) &&

	    (SEQ_LT(tp->snd_wl1, ti->ti_seq) ||

	    (tp->snd_wl1 == ti->ti_seq && (SEQ_LT(tp->snd_wl2, ti->ti_ack) ||

	    (tp->snd_wl2 == ti->ti_ack && tiwin > tp->snd_wnd))))) {

		tp->snd_wnd = tiwin;

		tp->snd_wl1 = ti->ti_seq;

		tp->snd_wl2 = ti->ti_ack;

		if (tp->snd_wnd > tp->max_sndwnd)

			tp->max_sndwnd = tp->snd_wnd;

		needoutput = 1;

	}



	/*

	 * Process segments with URG.

	 */

	if ((tiflags & TH_URG) && ti->ti_urp &&

	    TCPS_HAVERCVDFIN(tp->t_state) == 0) {

		/*

		 * This is a kludge, but if we receive and accept

		 * random urgent pointers, we'll crash in

		 * soreceive.  It's hard to imagine someone

		 * actually wanting to send this much urgent data.

		 */

		if (ti->ti_urp + so->so_rcv.sb_cc > so->so_rcv.sb_datalen) {

			ti->ti_urp = 0;

			tiflags &= ~TH_URG;

			goto dodata;

		}

		/*

		 * If this segment advances the known urgent pointer,

		 * then mark the data stream.  This should not happen

		 * in CLOSE_WAIT, CLOSING, LAST_ACK or TIME_WAIT STATES since

		 * a FIN has been received from the remote side.

		 * In these states we ignore the URG.

		 *

		 * According to RFC961 (Assigned Protocols),

		 * the urgent pointer points to the last octet

		 * of urgent data.  We continue, however,

		 * to consider it to indicate the first octet

		 * of data past the urgent section as the original

		 * spec states (in one of two places).

		 */

		if (SEQ_GT(ti->ti_seq+ti->ti_urp, tp->rcv_up)) {

			tp->rcv_up = ti->ti_seq + ti->ti_urp;

			so->so_urgc =  so->so_rcv.sb_cc +

				(tp->rcv_up - tp->rcv_nxt); /* -1; */

			tp->rcv_up = ti->ti_seq + ti->ti_urp;



		}

	} else

		/*

		 * If no out of band data is expected,

		 * pull receive urgent pointer along

		 * with the receive window.

		 */

		if (SEQ_GT(tp->rcv_nxt, tp->rcv_up))

			tp->rcv_up = tp->rcv_nxt;

dodata:



	/*

	 * If this is a small packet, then ACK now - with Nagel

	 *      congestion avoidance sender won't send more until

	 *      he gets an ACK.

	 */

	if (ti->ti_len && (unsigned)ti->ti_len <= 5 &&

	    ((struct tcpiphdr_2 *)ti)->first_char == (char)27) {

		tp->t_flags |= TF_ACKNOW;

	}



	/*

	 * Process the segment text, merging it into the TCP sequencing queue,

	 * and arranging for acknowledgment of receipt if necessary.

	 * This process logically involves adjusting tp->rcv_wnd as data

	 * is presented to the user (this happens in tcp_usrreq.c,

	 * case PRU_RCVD).  If a FIN has already been received on this

	 * connection then we just ignore the text.

	 */

	if ((ti->ti_len || (tiflags&TH_FIN)) &&

	    TCPS_HAVERCVDFIN(tp->t_state) == 0) {

		TCP_REASS(tp, ti, m, so, tiflags);

	} else {

		m_free(m);

		tiflags &= ~TH_FIN;

	}



	/*

	 * If FIN is received ACK the FIN and let the user know

	 * that the connection is closing.

	 */

	if (tiflags & TH_FIN) {

		if (TCPS_HAVERCVDFIN(tp->t_state) == 0) {

			/*

			 * If we receive a FIN we can't send more data,

			 * set it SS_FDRAIN

                         * Shutdown the socket if there is no rx data in the

			 * buffer.

			 * soread() is called on completion of shutdown() and

			 * will got to TCPS_LAST_ACK, and use tcp_output()

			 * to send the FIN.

			 */

			sofwdrain(so);



			tp->t_flags |= TF_ACKNOW;

			tp->rcv_nxt++;

		}

		switch (tp->t_state) {



	 	/*

		 * In SYN_RECEIVED and ESTABLISHED STATES

		 * enter the CLOSE_WAIT state.

		 */

		case TCPS_SYN_RECEIVED:

		case TCPS_ESTABLISHED:

		  if(so->so_emu == EMU_CTL)        /* no shutdown on socket */

		    tp->t_state = TCPS_LAST_ACK;

		  else

		    tp->t_state = TCPS_CLOSE_WAIT;

		  break;



	 	/*

		 * If still in FIN_WAIT_1 STATE FIN has not been acked so

		 * enter the CLOSING state.

		 */

		case TCPS_FIN_WAIT_1:

			tp->t_state = TCPS_CLOSING;

			break;



	 	/*

		 * In FIN_WAIT_2 state enter the TIME_WAIT state,

		 * starting the time-wait timer, turning off the other

		 * standard timers.

		 */

		case TCPS_FIN_WAIT_2:

			tp->t_state = TCPS_TIME_WAIT;

			tcp_canceltimers(tp);

			tp->t_timer[TCPT_2MSL] = 2 * TCPTV_MSL;

			break;



		/*

		 * In TIME_WAIT state restart the 2 MSL time_wait timer.

		 */

		case TCPS_TIME_WAIT:

			tp->t_timer[TCPT_2MSL] = 2 * TCPTV_MSL;

			break;

		}

	}



	/*

	 * Return any desired output.

	 */

	if (needoutput || (tp->t_flags & TF_ACKNOW)) {

		(void) tcp_output(tp);

	}

	return;



dropafterack:

	/*

	 * Generate an ACK dropping incoming segment if it occupies

	 * sequence space, where the ACK reflects our state.

	 */

	if (tiflags & TH_RST)

		goto drop;

	m_free(m);

	tp->t_flags |= TF_ACKNOW;

	(void) tcp_output(tp);

	return;



dropwithreset:

	/* reuses m if m!=NULL, m_free() unnecessary */

	if (tiflags & TH_ACK)

		tcp_respond(tp, ti, m, (tcp_seq)0, ti->ti_ack, TH_RST);

	else {

		if (tiflags & TH_SYN) ti->ti_len++;

		tcp_respond(tp, ti, m, ti->ti_seq+ti->ti_len, (tcp_seq)0,

		    TH_RST|TH_ACK);

	}



	return;



drop:

	/*

	 * Drop space held by incoming segment and return.

	 */

	m_free(m);

}
