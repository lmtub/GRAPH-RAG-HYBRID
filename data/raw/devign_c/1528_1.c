static coroutine_fn int quorum_co_flush(BlockDriverState *bs)

{

    BDRVQuorumState *s = bs->opaque;

    QuorumVoteVersion *winner = NULL;

    QuorumVotes error_votes;

    QuorumVoteValue result_value;

    int i;

    int result = 0;



    QLIST_INIT(&error_votes.vote_list);

    error_votes.compare = quorum_64bits_compare;



    for (i = 0; i < s->num_children; i++) {

        result = bdrv_co_flush(s->children[i]->bs);

        result_value.l = result;

        quorum_count_vote(&error_votes, &result_value, i);

    }



    winner = quorum_get_vote_winner(&error_votes);

    result = winner->value.l;



    quorum_free_vote_list(&error_votes);



    return result;

}
