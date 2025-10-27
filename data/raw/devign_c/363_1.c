static void v9fs_renameat(void *opaque)
{
    ssize_t err = 0;
    size_t offset = 7;
    V9fsPDU *pdu = opaque;
    V9fsState *s = pdu->s;
    int32_t olddirfid, newdirfid;
    V9fsString old_name, new_name;
    v9fs_string_init(&old_name);
    v9fs_string_init(&new_name);
    err = pdu_unmarshal(pdu, offset, "dsds", &olddirfid,
                        &old_name, &newdirfid, &new_name);
    if (err < 0) {
    if (name_is_illegal(old_name.data) || name_is_illegal(new_name.data)) {
        err = -ENOENT;
    v9fs_path_write_lock(s);
    err = v9fs_complete_renameat(pdu, olddirfid,
                                 &old_name, newdirfid, &new_name);
    v9fs_path_unlock(s);
    if (!err) {
        err = offset;
out_err:
    pdu_complete(pdu, err);
    v9fs_string_free(&old_name);
    v9fs_string_free(&new_name);