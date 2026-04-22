if re_check:
        # 再判定モード
        query = "query { allImages { id files { path } tags { id } } }"
        # ここを修正: call_GQL
        res = client.call_GQL(query)
        targets = res.get('allImages', [])
    else:
        # 通常モード
        query = """
        query GetUnprocessed($filter: ImageFilterType) {
          allImages(image_filter: $filter) {
            id
            files { path }
            tags { id }
          }
        }
        """
        variables = {
            "filter": {
                "tags": { "value": [m_id, n_id], "modifier": "NOT_IN" }
            }
        }
        # ここを修正: call_GQL
        res = client.call_GQL(query, variables)
        targets = res.get('allImages', [])

    # --- (中略) ---

        if set(current_tids) != set(final_tags):
            mutation = """
            mutation Update($id: ID!, $tags: [ID!]) {
              imageUpdate(input: { id: $id, tag_ids: $tags }) { id }
            }
            """
            # ここも修正: call_GQL
            client.call_GQL(mutation, {"id": img_id, "tags": final_tags})
            status = " (Updated)"
