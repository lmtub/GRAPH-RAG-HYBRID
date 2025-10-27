static void tftp_send_error(struct tftp_session *spt,

                            uint16_t errorcode, const char *msg,

                            struct tftp_t *recv_tp)

{

  struct sockaddr_in saddr, daddr;

  struct mbuf *m;

  struct tftp_t *tp;



  m = m_get(spt->slirp);



  if (!m) {

    goto out;

  }



  memset(m->m_data, 0, m->m_size);



  m->m_data += IF_MAXLINKHDR;

  tp = (void *)m->m_data;

  m->m_data += sizeof(struct udpiphdr);



  tp->tp_op = htons(TFTP_ERROR);

  tp->x.tp_error.tp_error_code = htons(errorcode);

  pstrcpy((char *)tp->x.tp_error.tp_msg, sizeof(tp->x.tp_error.tp_msg), msg);



  saddr.sin_addr = recv_tp->ip.ip_dst;

  saddr.sin_port = recv_tp->udp.uh_dport;



  daddr.sin_addr = spt->client_ip;

  daddr.sin_port = spt->client_port;



  m->m_len = sizeof(struct tftp_t) - 514 + 3 + strlen(msg) -

        sizeof(struct ip) - sizeof(struct udphdr);



  udp_output2(NULL, m, &saddr, &daddr, IPTOS_LOWDELAY);



out:

  tftp_session_terminate(spt);

}
