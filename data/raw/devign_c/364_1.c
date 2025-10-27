if_start(void)
{
	struct mbuf *ifm, *ifqt;
	DEBUG_CALL("if_start");
	if (if_queued == 0)
	   return; /* Nothing to do */
 again:
        /* check if we can really output */
        if (!slirp_can_output())
            return;
	/*
	 * See which queue to get next packet from
	 * If there's something in the fastq, select it immediately
	 */
	if (if_fastq.ifq_next != &if_fastq) {
		ifm = if_fastq.ifq_next;
	} else {
		/* Nothing on fastq, see if next_m is valid */
		if (next_m != &if_batchq)
		   ifm = next_m;
		else
		   ifm = if_batchq.ifq_next;
		/* Set which packet to send on next iteration */
		next_m = ifm->ifq_next;
	}
	/* Remove it from the queue */
	ifqt = ifm->ifq_prev;
	remque(ifm);
	--if_queued;
	/* If there are more packets for this session, re-queue them */
	if (ifm->ifs_next != /* ifm->ifs_prev != */ ifm) {
		insque(ifm->ifs_next, ifqt);
		ifs_remque(ifm);
	}
	/* Update so_queued */
	if (ifm->ifq_so) {
		if (--ifm->ifq_so->so_queued == 0)
		   /* If there's no more queued, reset nqueued */
		   ifm->ifq_so->so_nqueued = 0;
	}
	/* Encapsulate the packet for sending */
        if_encap(ifm->m_data, ifm->m_len);
	if (if_queued)
	   goto again;
}