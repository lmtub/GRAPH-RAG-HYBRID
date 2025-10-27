static void tftp_handle_rrq(Slirp *slirp, struct tftp_t *tp, int pktlen)

{

  struct tftp_session *spt;

  int s, k;

  size_t prefix_len;

  char *req_fname;



  /* check if a session already exists and if so terminate it */

  s = tftp_session_find(slirp, tp);

  if (s >= 0) {

    tftp_session_terminate(&slirp->tftp_sessions[s]);

  }



  s = tftp_session_allocate(slirp, tp);



  if (s < 0) {

    return;

  }



  spt = &slirp->tftp_sessions[s];



  /* unspecifed prefix means service disabled */

  if (!slirp->tftp_prefix) {

      tftp_send_error(spt, 2, "Access violation", tp);

      return;

  }



  /* skip header fields */

  k = 0;

  pktlen -= ((uint8_t *)&tp->x.tp_buf[0] - (uint8_t *)tp);



  /* prepend tftp_prefix */

  prefix_len = strlen(slirp->tftp_prefix);

  spt->filename = qemu_malloc(prefix_len + TFTP_FILENAME_MAX + 2);

  memcpy(spt->filename, slirp->tftp_prefix, prefix_len);

  spt->filename[prefix_len] = '/';



  /* get name */

  req_fname = spt->filename + prefix_len + 1;



  while (1) {

    if (k >= TFTP_FILENAME_MAX || k >= pktlen) {

      tftp_send_error(spt, 2, "Access violation", tp);

      return;

    }

    req_fname[k] = (char)tp->x.tp_buf[k];

    if (req_fname[k++] == '\0') {

      break;

    }

  }



  /* check mode */

  if ((pktlen - k) < 6) {

    tftp_send_error(spt, 2, "Access violation", tp);

    return;

  }



  if (memcmp(&tp->x.tp_buf[k], "octet\0", 6) != 0) {

      tftp_send_error(spt, 4, "Unsupported transfer mode", tp);

      return;

  }



  k += 6; /* skipping octet */



  /* do sanity checks on the filename */

  if (!strncmp(req_fname, "../", 3) ||

      req_fname[strlen(req_fname) - 1] == '/' ||

      strstr(req_fname, "/../")) {

      tftp_send_error(spt, 2, "Access violation", tp);

      return;

  }



  /* check if the file exists */

  if (tftp_read_data(spt, 0, NULL, 0) < 0) {

      tftp_send_error(spt, 1, "File not found", tp);

      return;

  }



  if (tp->x.tp_buf[pktlen - 1] != 0) {

      tftp_send_error(spt, 2, "Access violation", tp);

      return;

  }



  while (k < pktlen) {

      const char *key, *value;



      key = (const char *)&tp->x.tp_buf[k];

      k += strlen(key) + 1;



      if (k >= pktlen) {

	  tftp_send_error(spt, 2, "Access violation", tp);

	  return;

      }



      value = (const char *)&tp->x.tp_buf[k];

      k += strlen(value) + 1;



      if (strcmp(key, "tsize") == 0) {

	  int tsize = atoi(value);

	  struct stat stat_p;



	  if (tsize == 0) {

	      if (stat(spt->filename, &stat_p) == 0)

		  tsize = stat_p.st_size;

	      else {

		  tftp_send_error(spt, 1, "File not found", tp);

		  return;

	      }

	  }



	  tftp_send_oack(spt, "tsize", tsize, tp);

	  return;

      }

  }



  tftp_send_data(spt, 1, tp);

}
