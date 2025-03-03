# NeonAI Core LLM
Core module for Neon LLM's

## Request Format
API requests should include `history`, a list of tuples of strings, and the current
`query`

>Example Request:
>```json
>{
>  "history": [["user", "hello"], ["llm", "hi"]],
>  "query": "how are you?"
>}
>```

## Response Format
Responses will be returned as dictionaries. Responses should contain the following:
- `response` - String LLM response to the query

## Connection Configuration
When running this as a docker container, the `XDG_CONFIG_HOME` envvar is set to `/config`.
A configuration file at `/config/neon/diana.yaml` is required and should look like:
```yaml
MQ:
  port: <MQ Port>
  server: <MQ Hostname or IP>
  users:
    <LLM MQ service_name>:
      user: <MQ user>
      password: <MQ user's password>
  LLM_<LLM NAME uppercase>:
    num_parallel_processes: <integer > 0>
```

## Enabling Chatbot personas
An LLM may be configured to connect to a `/chatbots` vhost and participate in
discussions as described in the [chatbots project](https://github.com/NeonGeckoCom/chatbot-core).
One LLM may define multiple personas to participate as:
```yaml
llm_bots:
  <LLM Name>:
    - name: Assistant
      description: You are a personal assistant who responds in 40 words or less
    - name: Author
      description: You are an author and expert in literary history
    - name: Student
      description: You are a graduate student working in the field of artificial intelligence
      enabled: False
```
> `LLM Name` is defined in the property `NeonLLMMQConnector.name`
