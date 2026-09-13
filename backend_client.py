import os

from typing import Any, Optional
import httpx

class BackendClient:
    """HTTP client for the Spring Boot support platform API."""
    def __init__(self, jwt_token: str, base_url: Optional[str] = None):
        self.base_url = (base_url or os.getenv("BACKEND_URL", "http://localhost:8081")).rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Content-Type": "application/json",
        }

    def _unwrap(self, response: httpx.Response) -> Any:
        response.raise_for_status()
        body = response.json()
        if isinstance(body, dict) and "data" in body:
            return body["data"]
        return body

    def list_tickets(
        self,
        *,
        admin: bool,
        status: Optional[str] = None,
        title: Optional[str] = None,
        page: int = 0,
        size: int = 20,
        sort_by: str = "createdAt",
        sort_direction: str = "desc",

    ) -> dict:
        path = "/tickets/all" if admin else "/tickets/my-tickets"
        params: dict[str, Any] = {
            "page": page,
            "size": size,
            "sortBy": sort_by,
            "sortDirection": sort_direction,
        }
        if status:
            params["status"] = status
        if title:
            params["title"] = title

        with httpx.Client(timeout=30.0) as client:

            r = client.get(f"{self.base_url}{path}", headers=self.headers, params=params)

            return self._unwrap(r)


    def get_ticket(self, ticket_id: int) -> dict:
        with httpx.Client(timeout=30.0) as client:

            r = client.get(f"{self.base_url}/tickets/{ticket_id}", headers=self.headers)

            return self._unwrap(r)



    def create_ticket(self, payload: dict) -> dict:
        with httpx.Client(timeout=30.0) as client:

            r = client.post(f"{self.base_url}/tickets", headers=self.headers, json=payload)

            return self._unwrap(r)

    def update_ticket(self, ticket_id: int, payload: dict) -> dict:
        with httpx.Client(timeout=30.0) as client:

            r = client.put(

                f"{self.base_url}/tickets/{ticket_id}",

                headers=self.headers,

                json=payload,

            )

            return self._unwrap(r)



    def delete_ticket(self, ticket_id: int) -> None:
        with httpx.Client(timeout=30.0) as client:

            r = client.delete(f"{self.base_url}/tickets/{ticket_id}", headers=self.headers)

            r.raise_for_status()

