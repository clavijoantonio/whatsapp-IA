const { createBot, createProvider, createFlow, addKeyword } = require('@bot-whatsapp/bot')
const QRPortalWeb = require('@bot-whatsapp/portal')
const BaileysProvider = require('@bot-whatsapp/provider/baileys')
const MockAdapter = require('@bot-whatsapp/database/mock')
const io = require('socket.io-client');

// Configurar Socket.io
const socket = io('http://localhost:5000');

// Flujo de saludo inicial ESPECÍFICO
const flowInicial = addKeyword(['hola', 'hi', 'hello', 'buenas', 'saludos'])
  .addAction(async (ctx, { flowDynamic, state }) => {
    console.log('\n🎉 FLUJO INICIAL ACTIVADO');
    console.log('💬 Mensaje:', ctx.body);
    console.log('👤 From:', ctx.from);
    
    // Inicializar contexto
    const newContext = {
      history: [{
        user: ctx.body,
        timestamp: new Date().toISOString()
      }],
      lastInteraction: 'user',
      startedAt: new Date().toISOString(),
      isFirstMessage: false
    };
    
    await state.update({ context: newContext });
    console.log('🆕 Contexto inicializado');
    
    // ENVIAR RESPUESTA INMEDIATA DE BIENVENIDA
    await flowDynamic('🙌 ¡Hola! Bienvenido/a. ¿En qué puedo ayudarte hoy?');
    console.log('✅ Bienvenida enviada');
    
    // Actualizar contexto con la respuesta
    newContext.history.push({
      bot: '🙌 ¡Hola! Bienvenido/a. ¿En qué puedo ayudarte hoy?',
      timestamp: new Date().toISOString()
    });
    newContext.lastInteraction = 'bot';
    
    await state.update({ context: newContext });
    console.log('✅ Contexto actualizado con respuesta\n');
  });

// Flujo para mensajes subsiguientes
const chatHandler = addKeyword([])
  .addAction(
    { capture: true },
    async (ctx, { flowDynamic, state, gotoFlow }) => {
      try {
        console.log('\n=== MENSAJE SECUNDARIO ===');
        console.log('💬 Mensaje:', ctx.body);
        console.log('👤 From:', ctx.from);
        
        // Obtener contexto
        let context = await state.get('context');
        console.log('📊 Contexto existe:', !!context);
        
        // Si no hay contexto, podría ser un saludo que no capturó flowInicial
        if (!context) {
          console.log('⚠️  Sin contexto, verificando si es saludo...');
          const isGreeting = ['hola', 'hi', 'hello', 'buenas', 'saludos'].includes(ctx.body.toLowerCase());
          if (isGreeting) {
            console.log('🔁 Redirigiendo a flowInicial...');
            return gotoFlow(flowInicial);
          }
        }
        
        // Si hay contexto pero es el primer mensaje, también redirigir
        if (context && context.isFirstMessage !== false) {
          console.log('🔁 Contexto marcado como primer mensaje, redirigiendo...');
          return gotoFlow(flowInicial);
        }

        // Actualizar contexto
        context.history.push({
          user: ctx.body,
          timestamp: new Date().toISOString()
        });
        context.lastInteraction = 'user';
        
        await state.update({ context });
        console.log('✅ Contexto actualizado con mensaje usuario');

        // Verificar conexión
        if (!socket.connected) {
          console.log('❌ Socket no conectado');
          await flowDynamic('🔌 Problemas de conexión...');
          return null;
        }

        // Enviar al servidor Python
        console.log('📨 Enviando al servidor Python...');
        socket.emit('message', {
          text: ctx.body,
          context: context
        });

        // Esperar respuesta
        const respuesta = await new Promise((resolve, reject) => {
          const timeout = setTimeout(() => reject(new Error('Timeout')),40000);
          
          socket.once('response', (data) => {
            clearTimeout(timeout);
            resolve(data);
          });
        });

        console.log('✅ Respuesta recibida del servidor:', respuesta.text);

        // Actualizar contexto
        context.history.push({
          bot: respuesta.text,
          timestamp: new Date().toISOString()
        });
        context.lastInteraction = 'bot';
        
        await state.update({ context });
        console.log('✅ Contexto actualizado con respuesta bot');

        // ENVIAR RESPUESTA AL USUARIO
        console.log('📝 Enviando respuesta al usuario...');
        await flowDynamic(respuesta.text);
        console.log('✅ Respuesta enviada al usuario\n');

      } catch (error) {
        console.error('❌ Error:', error.message);
        await flowDynamic('😔 Ocurrió un error. Intenta nuevamente.');
      }
      
      return null;
    }
  );

const main = async () => {
  try {
    const adapterDB = new MockAdapter()
    
    // ORDEN CRÍTICO: flowInicial primero para capturar saludos
    const adapterFlow = createFlow([flowInicial, chatHandler])
    
    const adapterProvider = createProvider(BaileysProvider)
    
    createBot({
      flow: adapterFlow,
      provider: adapterProvider,
      database: adapterDB,
    })
    
    console.log('🚀 Bot iniciado - FlowInicial primero');
    QRPortalWeb()
    
  } catch (error) {
    console.error('❌ Error al iniciar:', error);
  }
}

// Logs de Socket.io
socket.on('connect', () => console.log('✅ Conectado al servidor Python'));
socket.on('disconnect', () => console.log('❌ Desconectado del servidor'));
socket.on('connect_error', (error) => console.log('❌ Error conexión:', error.message));

process.on('SIGINT', () => {
  console.log('\n🛑 Cerrando bot...');
  process.exit(0);
});

main()


    




