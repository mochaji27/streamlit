mkdir -p ~/.streamlit/
echo "\
[general]\n\n
email = \"muhamadaji1020@gmail.com\"\n\
" > ~/.streamlit/credential.toml
echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml