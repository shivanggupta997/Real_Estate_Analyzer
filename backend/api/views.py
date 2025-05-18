from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .processor import process_query

class ChatbotView(APIView):
    def post(self, request, *args, **kwargs):
        query = request.data.get('query')
        if not query:
            return Response({"error": "Query not provided"}, status=status.HTTP_400_BAD_REQUEST)

      
        result = process_query(query)

        return Response(result, status=status.HTTP_200_OK)