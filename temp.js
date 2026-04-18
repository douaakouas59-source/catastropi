
window.sendChat = async () => {
    const inp = document.getElementById('c-bot-input');
    const msg = inp.value.trim();
    if(!msg) return;
    
    const msgs = document.getElementById('c-bot-msgs');
    msgs.innerHTML += `<div style="background:var(--primary); color:white; padding:10px 15px; border-radius:15px; align-self:flex-end; max-width:80%; font-size:14px;">${msg}</div>`;
    inp.value = '';
    msgs.scrollTo(0, msgs.scrollHeight);
    
    try {
        const r = await api('/chatbot', 'POST', {query: msg});
        let respText = r.response.replace(/\\n/g, '<br>');
        msgs.innerHTML += `<div style="background:#e2e2e2; padding:10px 15px; border-radius:15px; align-self:flex-start; max-width:80%; font-size:14px;">${respText}</div>`;
        msgs.scrollTo(0, msgs.scrollHeight);
    } catch(e) {
        msgs.innerHTML += `<div style="background:#ffdada; color:red; padding:10px 15px; border-radius:15px; align-self:flex-start; max-width:80%; font-size:14px;">Erreur de connexion a l'IA.</div>`;
    }
}
