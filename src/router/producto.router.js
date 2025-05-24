const express = require('express');
const router = express.Router(); 
const productoController = require('../controller/producto.Controller');

router
    .get('/', productoController.get )
    .get('/:id', productoController.getById )
    .post('/', productoController.create )
    .put('/:id', productoController.update )
    .delete('/:id', productoController._delete );

module.exports = router;