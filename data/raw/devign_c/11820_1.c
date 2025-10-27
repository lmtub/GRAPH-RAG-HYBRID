static void v9fs_version(void *opaque)
{
    V9fsPDU *pdu = opaque;
    V9fsState *s = pdu->s;
    V9fsString version;
    size_t offset = 7;
    pdu_unmarshal(pdu, offset, "ds", &s->msize, &version);
    trace_v9fs_version(pdu->tag, pdu->id, s->msize, version.data);
    if (!strcmp(version.data, "9P2000.u")) {
        s->proto_version = V9FS_PROTO_2000U;
    } else if (!strcmp(version.data, "9P2000.L")) {
        s->proto_version = V9FS_PROTO_2000L;
    } else {
        v9fs_string_sprintf(&version, "unknown");
    }
    offset += pdu_marshal(pdu, offset, "ds", s->msize, &version);
    complete_pdu(s, pdu, offset);
    v9fs_string_free(&version);
    return;
}