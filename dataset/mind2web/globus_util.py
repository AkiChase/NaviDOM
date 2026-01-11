import json

import globus_sdk
from globus_sdk import NativeAppAuthClient, RefreshTokenAuthorizer, TransferClient
from globus_sdk.scopes import TransferScopes
from loguru import logger

CLIENT_ID = "1b9bac10-13a7-49eb-9b6b-a905113967bd"


def auth_save():
    auth_client = globus_sdk.NativeAppAuthClient(CLIENT_ID)
    auth_client.oauth2_start_flow(requested_scopes=TransferScopes.all, refresh_tokens=True)
    authorize_url = auth_client.oauth2_get_authorize_url()
    print(f"Please go to this URL and login:\n\n{authorize_url}\n")
    auth_code = input("Please enter the code here: ").strip()
    tokens = auth_client.oauth2_exchange_code_for_tokens(auth_code)
    # tokens = auth_client.oauth2_exchange_code_for_tokens(auth_code)
    with open("globus_tokens.json", "w") as f:
        json.dump(tokens.by_resource_server["transfer.api.globus.org"], f, indent=4)


def auth_load():
    auth_client = NativeAppAuthClient(CLIENT_ID)
    with open("globus_tokens.json", "r") as f:
        saved_tokens = json.load(f)

    authorizer = RefreshTokenAuthorizer(refresh_token=saved_tokens["refresh_token"], auth_client=auth_client)

    transfer_client = TransferClient(authorizer=authorizer)
    return transfer_client


def globus_download(transfer_client: TransferClient, src_id: str, dest_id: str, data_items: list[tuple[str, str]]):
    task_data = globus_sdk.TransferData(source_endpoint=src_id, destination_endpoint=dest_id)

    for item in data_items:
        task_data.add_item(*item)

    task_doc = transfer_client.submit_transfer(task_data)
    task_id = task_doc["task_id"]
    logger.info(f"submitted transfer, task_id={task_id}")
